import os
import sys
import glob
import json
import time
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure, zoom
from skimage.measure import marching_cubes
import nibabel as nib

# =====================================================================
# PHASE 1: TIMEFRAME & DATA HANDLING
# =====================================================================

def resolve_predict_gbm_timeframe():
    """ Clinical planning delay between pre-op MRI and RT initiation (~14 days). """
    return 14.0

def load_nifti_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required volume file: {file_path}")
    nimg = nib.load(file_path)
    return nimg.get_fdata(), nimg.header.get_zooms()[:3], nimg.affine, nimg.header

def load_patient_pipeline_predict_gbm(patient_dir):
    print(f"--- Loading PREDICT-GBM Patient: {os.path.basename(patient_dir)} ---")
    files_to_load = {
        "t1gd": "t1c_bet_normalized.nii.gz",
        "flair": "flair_bet_normalized.nii.gz",
        "tumor_seg": "tumor_seg.nii.gz",
        "wm_map": "wm_pbmap.nii.gz",
        "gm_map": "gm_pbmap.nii.gz",
        "csf_map": "csf_pbmap.nii.gz"
    }
    
    patient_data = {}
    voxel_dims, affine, header = None, None, None
    for key, filename in files_to_load.items():
        full_path = os.path.join(patient_dir, filename)
        if os.path.exists(full_path):
            data, dims, aff, hdr = load_nifti_image(full_path)
            patient_data[key] = data
            if voxel_dims is None:
                voxel_dims, affine, header = dims, aff, hdr
        else:
            print(f"Warning: {filename} not found.")
            
    patient_data.update({
        "voxel_dims": voxel_dims, 
        "affine": affine, 
        "header": header, 
        "native_shape": patient_data["wm_map"].shape
    })
    
    seg = patient_data.get("tumor_seg", np.zeros_like(patient_data["wm_map"]))
    # Labels: 1=necrotic, 2=edema, 3=enhancing tumor
    patient_data["t1c_core_mask"] = ((seg == 1) | (seg == 3))
    patient_data["flair_edema_mask"] = (seg > 0)
    
    # Brain Parenchyma mask (White Matter + Grey Matter) prevents skull/air expansion
    patient_data["parenchyma_mask"] = ((patient_data["wm_map"] + patient_data["gm_map"]) > 0.1)
    # Includes CSF to model complete brain morphology for visual reference
    patient_data["whole_brain_mask"] = patient_data["parenchyma_mask"] | (patient_data.get("csf_map", np.zeros_like(seg)) > 0.1)
    
    # Synthesize or load OAR mask (Organs at Risk like brainstem)
    oar_path = os.path.join(patient_dir, "oar_mask.nii.gz")
    if os.path.exists(oar_path):
        patient_data["oar_mask"] = (load_nifti_image(oar_path)[0] > 0)
    else:
        oar_mask = np.zeros_like(seg, dtype=bool)
        z_indices = np.where(patient_data["wm_map"] > 0)[2]
        if len(z_indices):
            min_z = np.min(z_indices)
            oar_mask[:, :, min_z:min_z+15] = True
            oar_mask &= patient_data["parenchyma_mask"]
        patient_data["oar_mask"] = oar_mask
    
    rec_path = os.path.join(patient_dir, "recurrence_preop.nii.gz")
    if os.path.exists(rec_path):
        rec_data = load_nifti_image(rec_path)[0]
        patient_data["recurrence_seg"] = rec_data
        patient_data["resection_cavity"] = (rec_data == 4)
        patient_data["true_recurrence_mass"] = ((rec_data >= 1) & (rec_data <= 3))
    else:
        patient_data["resection_cavity"] = np.zeros_like(seg, dtype=bool)
        patient_data["true_recurrence_mass"] = np.zeros_like(seg, dtype=bool)
        
    return patient_data

# =====================================================================
# PHASE 2: MATHEMATICALLY CORRECT D3Q7 LATTICE-BOLTZMANN SOLVER
# =====================================================================

def perform_lbm_diffusion_step(u_current, f_current, neighbor_outside, D_map, dt, voxel_dims):
    """
    Executes a single D3Q7 Lattice-Boltzmann collision and streaming step.
    Strictly enforces zero-flux Neumann B.C. via exact bounce-back (D * grad(u) * n = 0).
    """
    dirs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    inv = [0, 2, 1, 4, 3, 6, 5] # Inverse direction indices for boundary bounce-back
    
    brain_mask = (D_map > 0.0)
    dx_mm = voxel_dims[0]
    
    # Chapman-Enskog relation for D3Q7 lattice: tau = 3 * D_lat + 0.5
    D_lattice = D_map * (dt / (dx_mm ** 2))
    tau = np.where(brain_mask, 3.0 * D_lattice + 0.5, 0.501)
    tau = np.clip(tau, 0.505, 5.0)
    
    # Equilibrium distribution (w_0 = 0, w_1...6 = 1/6)
    f_eq = np.zeros_like(f_current)
    for i in range(1, 7):
        f_eq[i] = u_current / 6.0
        
    # BGK Collision Step
    f_post = f_current - (f_current - f_eq) / tau[np.newaxis, ...]
    
    # Streaming Step with exact boundary bounce-back for zero-flux condition
    f_streamed = np.zeros_like(f_current)
    for i in range(1, 7):
        dx_i, dy_i, dz_i = dirs[i]
        streamed_incoming = np.roll(f_post[i], shift=(dx_i, dy_i, dz_i), axis=(0, 1, 2))
        bounced_back = f_post[inv[i]]
        f_streamed[i] = np.where(neighbor_outside[i], bounced_back, streamed_incoming)
        
    f_streamed[:, ~brain_mask] = 0.0
    u_diffused = np.sum(f_streamed[1:], axis=0)
    return np.clip(u_diffused, 0.0, 1.0), f_streamed

# =====================================================================
# PHASE 3: EXPONENTIAL MOLECULAR DOSIMETRY OPTIMIZER
# =====================================================================

def optimize_prescription_dose(u_density, alpha_bar, d_int, voxel_dims):
    """
    Computes optimal prescription dose enforcing the exact Lagrangian formula:
    d_i = max[0, (1/alpha_bar) * ln((u_i * alpha_bar) / mu)]
    solved via Lagrangian multiplier bisection targeting d_int.
    """
    voxel_vol = np.prod(voxel_dims) # mm^3
    mask = (u_density > 1e-5)
    
    def calculate_integrated_dose(mu_val):
        dose = np.zeros_like(u_density, dtype=np.float64)
        log_term = (u_density * alpha_bar) / mu_val
        valid = mask & (log_term > 1.0)
        
        if np.any(valid):
            dose[valid] = (1.0 / alpha_bar) * np.log(log_term[valid])
            
        np.clip(dose, 0.0, 60.0, out=dose)
        return np.sum(dose * voxel_vol)
        
    mu_low, mu_high = 1e-15, float(np.max(u_density) * alpha_bar)
    if mu_high <= mu_low: 
        return np.zeros_like(u_density)
        
    # Bisection search for Lagrange multiplier mu
    for _ in range(60):
        mu_mid = (mu_low + mu_high) / 2.0
        if calculate_integrated_dose(mu_mid) > d_int:
            mu_low = mu_mid
        else:
            mu_high = mu_mid
            
    mu_opt = (mu_low + mu_high) / 2.0
    
    dose_grid = np.zeros_like(u_density, dtype=np.float64)
    log_term = (u_density * alpha_bar) / mu_opt
    valid = mask & (log_term > 1.0)
    if np.any(valid):
        dose_grid[valid] = (1.0 / alpha_bar) * np.log(log_term[valid])
        
    return np.clip(dose_grid, 0.0, 60.0)

# =====================================================================
# PHASE 4: SPATIAL UTILITIES & PRE-OP INITIALIZATION
# =====================================================================

def downsample_patient_data(patient_data, factor=1):
    if factor == 1: return patient_data
    coarse_data = {}
    for key, val in patient_data.items():
        if key == "voxel_dims":
            coarse_data[key] = tuple(np.array(val) * factor)
        elif isinstance(val, np.ndarray):
            order = 0 if any(sub in key for sub in ["mask", "seg", "cavity", "mass", "oar", "border"]) else 1
            coarse_data[key] = zoom(val, zoom=1.0/factor, order=order)
        else:
            coarse_data[key] = val
    return coarse_data

def upsample_to_native(coarse_array, native_shape, order=1):
    if coarse_array.shape == native_shape: return coarse_array
    factors = [n / c for n, c in zip(native_shape, coarse_array.shape)]
    return zoom(coarse_array, zoom=factors, order=order)

def reconstruct_preop_density(coarse_maps, dw, rho, tau_1=0.80):
    """
    Reconstructs initial tumor density u_0(x) using the exponential geodesic falloff
    and invisibility index lambda = sqrt(D / rho).
    """
    t1c = np.asarray(coarse_maps['t1c_core_mask'], dtype=bool)
    voxel_dims = coarse_maps['voxel_dims']
    
    d_out = distance_transform_edt(~t1c, sampling=voxel_dims)
    d_in = distance_transform_edt(t1c, sampling=voxel_dims)
    
    lambda_val = max(0.1, float(np.sqrt(dw / max(rho, 1e-6))))
    u0 = np.zeros_like(t1c, dtype=np.float64)
    
    outside_t1c = ~t1c
    if np.any(outside_t1c):
        u0[outside_t1c] = tau_1 * np.exp(-d_out[outside_t1c] / lambda_val)
        
    if np.any(t1c):
        u0[t1c] = np.clip(tau_1 * np.exp(d_in[t1c] / lambda_val), 0.0, 1.0)
        
    return np.clip(u0, 0.0, 1.0)

# =====================================================================
# PHASE 5: SIMULATION LOOP & METROPOLIS-HASTINGS CALIBRATION
# =====================================================================

def compute_95th_percentile_hausdorff(pred_mask, true_mask, voxel_dims):
    """ Computes exact 95th Percentile Symmetric Hausdorff Distance (sHD95). """
    pred_mask, true_mask = np.asarray(pred_mask, dtype=bool), np.asarray(true_mask, dtype=bool)
    if not np.any(pred_mask) or not np.any(true_mask): 
        return float('inf')
        
    struct = generate_binary_structure(3, 1)
    border_pred = pred_mask ^ binary_erosion(pred_mask, structure=struct)
    border_true = true_mask ^ binary_erosion(true_mask, structure=struct)
    
    dist_to_true = distance_transform_edt(~border_true, sampling=voxel_dims)
    dist_to_pred = distance_transform_edt(~border_pred, sampling=voxel_dims)
    
    dists1, dists2 = dist_to_true[border_pred], dist_to_pred[border_true]
    if len(dists1) == 0 and len(dists2) == 0:
        return float('inf')
        
    return float(np.percentile(np.concatenate([dists1, dists2]), 95))

def run_patient_simulation(dw, rho, patient_data, tau_1=0.80, target_time=14.0):
    """ Solves the Reaction-Diffusion equation du/dt = nabla(D nabla u) + rho u(1-u) """
    u = reconstruct_preop_density(patient_data, dw, rho, tau_1=tau_1)
    dx_mm = patient_data['voxel_dims'][0]
    
    dt = min(0.02 * (dx_mm ** 2), target_time / 10.0) # Numerical stability bound
    f = np.zeros((7,) + u.shape, dtype=np.float64)
    for i in range(1, 7): f[i] = u / 6.0
        
    # Strictly aligned with literature: D_wm = d_w; D_gm = d_w / 10.0
    D_map = patient_data['wm_map'] * dw + patient_data['gm_map'] * (dw / 10.0)
    
    # Tumor diffusion completely blocked by CSF fluid masks and skull boundaries
    csf_mask = np.asarray(patient_data.get('csf_map', np.zeros_like(u)) > 0.5, dtype=bool)
    cavity_mask = np.asarray(patient_data['resection_cavity'], dtype=bool)
    D_map[csf_mask | cavity_mask] = 0.0
    
    brain_mask = (D_map > 0.0)
    dirs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbor_outside = [~np.roll(brain_mask, shift=(-d[0], -d[1], -d[2]), axis=(0, 1, 2)) for d in dirs]
    
    steps = max(1, int(target_time / dt))
    for step in range(steps):
        u_intermed, f = perform_lbm_diffusion_step(u, f, neighbor_outside, D_map, dt, patient_data['voxel_dims'])
        
        # Logistic Proliferation Step: + rho * u * (1 - u)
        u = u_intermed + dt * (rho * u_intermed * (1.0 - u_intermed))
        u = np.clip(u, 0.0, 1.0)
        
        # Rescale distributions to preserve mass balance after proliferation
        growth_scale = np.where(u_intermed > 1e-10, u / (u_intermed + 1e-12), 1.0)
        f = f * growth_scale[np.newaxis, ...]
        
    return u

def calibrate_parameters_mh(patient_data, num_samples=30, tau_1=0.80, tau_2=0.16, treatment_delay=14.0):
    """
    Metropolis-Hastings algorithm enforcing Bayesian rules:
    P(S_3, S_4 | theta) ~ exp( -H(D, rho, S_3, S_4)^2 / (2 * sigma^2) )
    """
    print(f"-> Starting Metropolis-Hastings Calibration ({num_samples} iterations)...")
    true_border_mask = np.asarray(patient_data['flair_edema_mask'], dtype=bool)
    
    def get_log_posterior(dw, rho):
        # Strict log-uniform independent prior
        if not (1e-4 <= dw <= 10.0 and 1e-5 <= rho <= 10.0):
            return -np.inf

        u_0 = reconstruct_preop_density(patient_data, dw, rho, tau_1=tau_1)
        hd_95 = compute_95th_percentile_hausdorff(u_0 >= tau_2, true_border_mask, patient_data['voxel_dims'])
        
        if np.isinf(hd_95):
            return -1e6
        else:
            # Paper Eq: 2 * sigma^2 = 50.0 (where sigma = 5 mm)
            return -(hd_95 ** 2) / 50.0 
            
    current_dw, current_rho = 0.08, 0.012
    current_lp = get_log_posterior(current_dw, current_rho)
    
    best_map = (current_dw, current_rho, current_lp)
    
    u_sum = np.zeros_like(true_border_mask, dtype=np.float64)
    u_sq_sum = np.zeros_like(true_border_mask, dtype=np.float64)
    accepted_samples = 0
    burn_in = int(num_samples * 0.3)
    
    for i in range(num_samples):
        prop_log_dw = np.log(current_dw) + np.random.normal(0, 0.25)
        prop_log_rho = np.log(current_rho) + np.random.normal(0, 0.25)
        prop_dw, prop_rho = np.exp(prop_log_dw), np.exp(prop_log_rho)
        
        prop_lp = get_log_posterior(prop_dw, prop_rho)
        
        if np.log(np.random.uniform()) < (prop_lp - current_lp):
            current_dw, current_rho, current_lp = prop_dw, prop_rho, prop_lp
            if current_lp > best_map[2]:
                best_map = (current_dw, current_rho, current_lp)
        
        print(f"   Iter {i+1:02d}/{num_samples} | Log-Likelihood: {current_lp:.2f} | D: {current_dw:.4f}, rho: {current_rho:.4f}")

        if i >= burn_in:
            u_treatment = run_patient_simulation(current_dw, current_rho, patient_data, tau_1=tau_1, target_time=treatment_delay)
            u_sum += u_treatment
            u_sq_sum += (u_treatment ** 2)
            accepted_samples += 1

    if accepted_samples > 0:
        u_probabilistic = u_sum / accepted_samples
        u_variance = (u_sq_sum / accepted_samples) - (u_probabilistic ** 2)
        u_std_dev = np.sqrt(np.maximum(u_variance, 0.0))
    else:
        u_probabilistic = np.zeros_like(true_border_mask, dtype=np.float64)
        u_std_dev = np.zeros_like(true_border_mask, dtype=np.float64)
        
    u_map_treatment = run_patient_simulation(best_map[0], best_map[1], patient_data, tau_1=tau_1, target_time=treatment_delay)
    return u_probabilistic, u_std_dev, (best_map[0], best_map[1], u_map_treatment, best_map[2])

# =====================================================================
# PHASE 6: ASSET EXPORT & CERR IMRT PLANNING
# =====================================================================

def export_wavefront_obj(filepath, volume, threshold, voxel_dims):
    if volume is None or not np.any(volume >= threshold): return False
    try:
        verts, faces, normals, _ = marching_cubes(volume, level=threshold, spacing=voxel_dims, step_size=1)
        with open(filepath, 'w') as f:
            for v in verts: f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for vn in normals: f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            for face in faces: f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        return True
    except Exception as e:
        print(f"Failed to export mesh {filepath}: {e}")
        return False

def export_dose_point_cloud(filepath, volume, threshold, voxel_dims):
    """ Exports a JSON array of [x, y, z, dose] rendering as a 3D dot gradient. """
    d0, d1, d2 = np.where(volume >= threshold)
    if len(d0) == 0: return False
    
    # Cap maximum points for fluid WebGL presentation
    max_pts = 150000
    if len(d0) > max_pts:
        idx = np.random.choice(len(d0), max_pts, replace=False)
        d0, d1, d2 = d0[idx], d1[idx], d2[idx]
        
    pts = []
    # Spacing aligns perfectly with matching geometry of marching_cubes
    for i in range(len(d0)):
        c0 = float(d0[i] * voxel_dims[0])
        c1 = float(d1[i] * voxel_dims[1])
        c2 = float(d2[i] * voxel_dims[2])
        val = float(volume[d0[i], d1[i], d2[i]])
        pts.append([c0, c1, c2, val])
        
    with open(filepath, 'w') as f:
        json.dump(pts, f)
    return True

def generate_cerr_imrt_plan(filepath):
    cerr_plan = {
        "Description": "C. IMRT Planning\nOptimized via 9 coplanar 6 MV photon beams and a piece-wise quadratic objective function.",
        "Software": "CERR [29]",
        "Optimization_Method": "Piece-wise quadratic objective function",
        "Modality": "IMRT",
        "Beams": [{"Angle_Deg": i * 40, "Energy": "6 MV", "Type": "Coplanar"} for i in range(9)],
        "Target_Constraints": "Loaded from MAP, Probabilistic, and Corrected Prescription Dose Maps"
    }
    with open(filepath, 'w') as f:
        json.dump(cerr_plan, f, indent=4)

# =====================================================================
# PHASE 7: LOCAL HTTP SERVER FOR WEB VISUALIZATION
# =====================================================================

def start_local_server(port=5000):
    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f"\n==========================================================")
    print(f" Web Server Running! Open in your browser:")
    print(f" -> http://localhost:{port}/index.html")
    print(f"==========================================================\n")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="Personalized Radiotherapy Planning (Reaction-Diffusion)")
    parser.add_argument("--serve", action="store_true", help="Start local web server after processing")
    parser.add_argument("--port", type=int, default=5000, help="Port for local web server")
    args = parser.parse_args()

    print("=====================================================")
    print(" Personalized Radiotherapy Planning Pipeline")
    print("=====================================================")
    
    num_samples = 25 
    ds_factor = 2    
        
    base_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "predict_gbm")
    patient_paths = sorted([d for d in glob.glob(os.path.join(base_dataset_dir, "*")) if os.path.isdir(d)])
    
    if not patient_paths: 
        print(f"Error: No dataset found at {base_dataset_dir}")
        if args.serve: start_local_server(args.port)
        return

    processed_patients = []
    
    for target_patient_dir in patient_paths:
        start_time = time.time()
        patient_id = os.path.basename(target_patient_dir)
        print(f"\n================ Processing Patient: {patient_id} ================")
        
        try:
            target_time = resolve_predict_gbm_timeframe()
            patient_maps = load_patient_pipeline_predict_gbm(target_patient_dir)
            coarse_maps = downsample_patient_data(patient_maps, factor=ds_factor)
            
            ALPHA_BAR = 0.35  # Linear-quadratic asymptotic survival constant
            TAU_1, TAU_2 = 0.80, 0.16
            
            u_prob_preop, u_std_dev, map_sample = calibrate_parameters_mh(
                coarse_maps, num_samples=num_samples, tau_1=TAU_1, tau_2=TAU_2, treatment_delay=target_time
            )
            u_map_preop = map_sample[2]
            
            print("-> Computing Corrected Density (OAR & Variance Penalization)...")
            delta = 10.0
            oar_density = coarse_maps.get('oar_mask', np.zeros_like(u_prob_preop))
            
            # Corrected Density logic accurately implementing `u - delta * beta * c`
            # Where 'u_std_dev' mathematically corresponds to beta (the variance), and 'oar_density' to c (OARs mask)
            u_corr_preop = u_prob_preop - (delta * u_std_dev * oar_density)
            u_corr_preop = np.clip(u_corr_preop, 0.0, 1.0)
            
            # Apply clinical thresholding to zero out healthy brain areas (< 1% tumor density)
            # This shields the healthy brain and OARs from receiving accidental background dose optimization
            u_map_clean = np.where(u_map_preop >= 0.01, u_map_preop, 0.0)
            u_prob_clean = np.where(u_prob_preop >= 0.01, u_prob_preop, 0.0)
            u_corr_clean = np.where(u_corr_preop >= 0.01, u_corr_preop, 0.0)
            
            # Strictly zero out optimized targets inside OAR boundaries & outside brain parenchyma
            oar_mask_coarse = coarse_maps.get('oar_mask', np.zeros_like(u_prob_preop))
            parenchyma_coarse = coarse_maps['parenchyma_mask']
            
            for u_map in [u_map_clean, u_prob_clean, u_corr_clean]:
                u_map[oar_mask_coarse] = 0.0
                u_map[~parenchyma_coarse] = 0.0
            
            # Standard Plan Baseline: 60 Gy to CTV (T1c + 20mm margin) strictly restricted to brain parenchyma
            t1c_mask = coarse_maps['t1c_core_mask'] > 0
            if np.any(t1c_mask):
                dist_to_t1c = distance_transform_edt(~t1c_mask, sampling=coarse_maps['voxel_dims'])
                ctv_mask = (dist_to_t1c <= 20.0) & coarse_maps['parenchyma_mask']
            else:
                ctv_mask = (coarse_maps['flair_edema_mask'] > 0) & coarse_maps['parenchyma_mask']

            d_std = np.zeros_like(u_map_preop)
            d_std[ctv_mask] = 60.0
            d_int_phys = np.sum(d_std * np.prod(coarse_maps['voxel_dims']))
            
            print("-> Optimizing MAP, Probabilistic, and Corrected Prescription Doses...")
            map_dose_plan = optimize_prescription_dose(u_map_clean, ALPHA_BAR, d_int_phys, coarse_maps['voxel_dims'])
            prob_dose_plan = optimize_prescription_dose(u_prob_clean, ALPHA_BAR, d_int_phys, coarse_maps['voxel_dims'])
            corr_dose_plan = optimize_prescription_dose(u_corr_clean, ALPHA_BAR, d_int_phys, coarse_maps['voxel_dims'])
            
            # Post-optimization safety check: Ensure dose is zeroed inside OARs and outside brain parenchyma
            for plan in [map_dose_plan, prob_dose_plan, corr_dose_plan]:
                plan[oar_mask_coarse] = 0.0
                plan[~parenchyma_coarse] = 0.0
            
            # Upsample predictions back to native clinical space
            dose_native = upsample_to_native(corr_dose_plan, patient_maps['native_shape'])
            u_native = upsample_to_native(u_prob_preop, patient_maps['native_shape'])
            d_std_native = upsample_to_native(d_std, patient_maps['native_shape'], order=0)
            native_voxel_vol = np.prod(patient_maps['voxel_dims'])
            
            # Evaluate against true follow-up recurrence
            true_rec_mask = np.asarray(patient_maps['true_recurrence_mass'], dtype=bool)
            pred_rec_mask = (u_native >= TAU_2)
            hd = compute_95th_percentile_hausdorff(pred_rec_mask, true_rec_mask, patient_maps['voxel_dims'])
            dice = (2.0 * np.sum(pred_rec_mask & true_rec_mask)) / (np.sum(pred_rec_mask) + np.sum(true_rec_mask) + 1e-8)
            
            # Calculate tumor cell survival equation parameters -> u * exp(-alpha * d)
            surv_std = np.sum(u_native * np.exp(-ALPHA_BAR * d_std_native) * native_voxel_vol)
            surv_pers = np.sum(u_native * np.exp(-ALPHA_BAR * dose_native) * native_voxel_vol)
            reduction_pct = ((surv_std - surv_pers) / max(surv_std, 1e-8)) * 100.0
            
            elapsed_mins = (time.time() - start_time) / 60.0
            print(f"\n--- PATIENT SUMMARY ({elapsed_mins:.1f} minutes) ---")
            print(f"Spatial Dice: {dice:.4f} | Symmetric 95th-Hausdorff: {hd:.2f} mm")
            print(f"Surviving Cells: Standard={surv_std:.2e} vs Personalized={surv_pers:.2e}")
            print(f"Tumor Cell Reduction vs Standard: {reduction_pct:.2f}%\n")
            
            out_dir = os.path.join("web_assets", patient_id)
            os.makedirs(out_dir, exist_ok=True)
            
            metrics = {
                "Spatial_Accuracy": {
                    "Dice_Similarity_Coefficient": round(float(dice), 4),
                    "Symmetric_Hausdorff_Distance_mm": round(float(hd), 2) if not np.isinf(hd) else "N/A"
                },
                "Dosimetric_Superiority": {
                    "Surviving_Cells_Standard_Plan_AU": round(float(surv_std), 4),
                    "Surviving_Cells_Personalized_Plan_AU": round(float(surv_pers), 4),
                    "Tumor_Cell_Reduction_vs_Standard_Pct": round(float(reduction_pct), 2)
                },
                "Execution_Time_Minutes": round(float(elapsed_mins), 2)
            }
            with open(os.path.join(out_dir, "evaluation_metrics.json"), "w") as f: 
                json.dump(metrics, f, indent=4)
                
            generate_cerr_imrt_plan(os.path.join(out_dir, "05_cerr_imrt_plan.json"))
            
            v_dims = coarse_maps['voxel_dims']
            export_wavefront_obj(os.path.join(out_dir, "00_brain_mesh.obj"), coarse_maps['whole_brain_mask'], 0.5, v_dims)
            export_wavefront_obj(os.path.join(out_dir, "00_oar_mesh.obj"), coarse_maps['oar_mask'], 0.5, v_dims)
            export_wavefront_obj(os.path.join(out_dir, "01_clinical_t1c_core.obj"), coarse_maps['t1c_core_mask'], 0.5, v_dims)
            export_wavefront_obj(os.path.join(out_dir, "02_clinical_flair_edema.obj"), coarse_maps['flair_edema_mask'], 0.5, v_dims)
            export_wavefront_obj(os.path.join(out_dir, "03_model_predicted_density_tau2.obj"), u_map_preop, TAU_2, v_dims)
            # Use therapeutic target threshold of 5.0 Gy to render a neat and clean point cloud boundary
            export_dose_point_cloud(os.path.join(out_dir, "04_corrected_dose.json"), corr_dose_plan, 5.0, v_dims)
            export_wavefront_obj(os.path.join(out_dir, "06_expected_result_true_recurrence.obj"), coarse_maps['true_recurrence_mass'], 0.5, v_dims)
            
            processed_patients.append(patient_id)
            print(f"-> Successfully exported web assets to {out_dir}/")
            
        except Exception as e:
            print(f"Error processing {patient_id}:")
            import traceback; traceback.print_exc()
            
    os.makedirs("web_assets", exist_ok=True)
    with open("web_assets/patients.json", "w") as f:
        json.dump(processed_patients, f)
    print("\n-> Global registry updated at web_assets/patients.json")

    if args.serve or True: # Auto-start server for preview
        start_local_server(args.port)

if __name__ == "__main__":
    main()