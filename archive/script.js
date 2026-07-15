const N = 60, M = 60;
const NAP = 3, CAP = 0.15, GAP = 0.5, HAP = 7.1, NPH = 7.4;
const H0 = 1.0e-13;
const K = 6, TR = 0.675;
const Q = 5;
const DT = 5e-4;

const TAU = 16 * 3600;
const N0 = 1.6e5, C0 = 1.7e-9/*1.7e-8 if not hypoxia*/, G0 = 1.3e-8;


const P = 0.01 * (400 / N) * (400 / M) /*16x because grid is 16x smaller than the one used in the paper*/,
      AP = 1.0,
      SIGMA = 0.25;

const R_C_BASE = 2.3e-16,
      R_G_AE_BASE = 3.8e-17,
      R_G_AN_BASE = 6.9e-16,
      R_H_BASE = 1.5e-18;

const R_C = (TAU * N0 * R_C_BASE) / C0,
      R_G_AE = (TAU * N0 * R_G_AE_BASE) / G0,
      R_G_AN = (TAU * N0 * R_G_AN_BASE) / G0,
      R_H = (TAU * N0 * R_H_BASE) / H0;

const DA = [-1, 1, 0, 0];
const DB = [0, 0, -1, 1];

// diffusion
const D_C = 1.0368;
const D_G = 5.2416;
const D_H = 0.6336;
const DX = 0.0025;

// CHANGED: Query and maintain individual contexts for all four unique target canvases
const canvasTumor = document.getElementById("tumorCanvas");
const ctxTumor = canvasTumor.getContext("2d");

const canvasOxygen = document.getElementById("oxygenCanvas");
const ctxOxygen = canvasOxygen.getContext("2d");

const canvasGlucose = document.getElementById("glucoseCanvas");
const ctxGlucose = canvasGlucose.getContext("2d");

const canvasAcid = document.getElementById("acidCanvas");
const ctxAcid = canvasAcid.getContext("2d");

const CELL_SIZE = canvasTumor.width / N; 

let l = 0, r = 0;

class Cell {
    constructor() {}

    set(x, y, a, b) {
        this.x = x;
        this.y = y;
        this.a = a;
        this.b = b;

        this.w = [[1, 0, 0, 0], [0.5, 0, 0, 0], [0, -2, 0, 0], [0, 0, -2, 0.5], [1, 0, 0, 0]];
        this.W = [[-0.5, 1, -0.5, 0, 0], [0, 0.55, -0.5, 0, 0], [0, 0, 2, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]];
        this.theta = [0.55, 0, 0.7, -0.25, 0];
        this.phi = [0, 0, 0, 0, 0.75];

        this.xi = [0, 1, 1, 1]; // [Neighbors, Oxygen, Glucose, Proton]
        this.age = 0;
        this.split_age = AP;
        this.status = "empty";
    }

    copy(c) {
        this.x = c.x;
        this.y = c.y;
        this.a = c.a;
        this.b = c.b;

        this.w = structuredClone(c.w);
        this.W = structuredClone(c.W);
        this.theta = structuredClone(c.theta);
        this.phi = structuredClone(c.phi);

        this.xi = structuredClone(c.xi); // [Neighbors, Oxygen, Glucose, Proton]
        this.age = c.age;
        this.split_age = c.split_age;
        this.status = c.status;
    }

    static T(x) {
        let res = Array.from({length: x.length}, () => 0);
        for (let i = 0; i < x.length; i++) {
            res[i] = 1 / (1 + Math.exp(-2 * x[i]));
        }
        return res;
    }

    static mul(m, v) {
        let res = [];
        for (let i = 0; i < m.length; i++) {
            let sum = 0;
            for (let j = 0; j < v.length; j++) {
                sum += m[i][j] * v[j];
            }
            res.push(sum);
        }
        return res;
    }

    static add(a, b) {
        let res = [];
        for (let i = 0; i < a.length; i++) {
            res[i] = a[i] + b[i];
        }
        return res;
    }

    static sub(a, b) {
        let res = [];
        for (let i = 0; i < a.length; i++) {
            res[i] = a[i] - b[i];
        }
        return res;
    }

    static get_u(o, f) {
        return o[3] >= 0.5 ? [0, -R_C * f * DT, -R_G_AE * f * DT, 0] : [0, 0, -R_G_AN * f * DT, R_H * f * DT];
    }

    static in_bounds(a, b) {
        return a >= 0 && a < N && b >= 0 && b < M;
    }

    static gen_poisson(lambda) {
        const L = Math.exp(-lambda);
        let k = 0;
        let p = 1.0;

        do {
            k++;
            p *= Math.random();
        } while (p > L);

        return k - 1;
    }

    static gen_normal(mean, stdDev) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); 
        while (v === 0) v = Math.random();
        
        const z0 = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z0 * stdDev + mean;
    }

    static mutate(c, p = P) {
        let nc = new Cell();
        nc.copy(c);

        let rnd = Cell.gen_poisson(p);
        let pool = [];
        for (let i = 0; i < nc.w.length; i++)
            for (let j = 0; j < nc.w[i].length; j++)
                pool.push({target: nc.w[i], key: j});
        for (let i = 0; i < nc.W.length; i++)
            for (let j = 0; j < nc.W[i].length; j++)
                pool.push({target: nc.W[i], key: j});
        for (let i = 0; i < nc.theta.length; i++)
            pool.push({target: nc.theta, key: i});
        for (let i = 0; i < nc.phi.length; i++)
            pool.push({target: nc.phi, key: i});
        for (let m = 0; m < rnd; m++) {
            if (pool.length === 0) break;
            let ridx = Math.floor(Math.random() * pool.length);
            let ref = pool[ridx];
            let v = ref.target[ref.key];
            ref.target[ref.key] = v + Cell.gen_normal(0, SIGMA);
        }

        nc.split_age = Cell.gen_normal(AP, AP / 2);

        return nc;
    }

    static get_neighbors(a, b) {
        let res = 0;
        for (let i = 0; i < DA.length; i++) {
            let na = a + DA[i], nb = b + DB[i];
            if (!Cell.in_bounds(na, nb)) {
                res++;
                continue;
            }
            if (y[na][nb].status == "necrosis" || y[na][nb].status == "alive") {
                res++;
            }
        }
        return res;
    }

    apoptosis() {
        // apoptosis
        this.status = "apoptosis";
        y[this.a][this.b] = this;

        for (let i = 0; i < DA.length; i++) {
            let na = this.a + DA[i], nb = this.b + DB[i];
            if (Cell.in_bounds(na, nb)) {
                y[na][nb].xi[0] = Cell.get_neighbors(na, nb);
            }
        }
    }

    necrosis() {
        // necrosis
        this.status = "necrosis";
        y[this.a][this.b] = this;
    }

    tick() {
        if (this.status != "alive") return;

        // xi: [Neighbors, Oxygen, Glucose, Acidity]
        // o: [Proliferation, Quiescence, Apoptosis, Metabolism, Movement]

        // neural network
        let res = new Cell();
        this.xi = structuredClone(y[this.a][this.b].xi);
        res.copy(this);
        let v = Cell.T(Cell.sub(Cell.mul(this.w, this.xi), this.theta));
        let o = Cell.T(Cell.sub(Cell.mul(this.W, v), this.phi));
        
        // mutually exclusive responses: proliferation, quiescence, apoptosis
        let max = Math.max(o[0], o[1], o[2]);
        
        if (o[3] < 0.5) l++;
        else r++;

        // cell death
        if (max == o[2]) {
            res.apoptosis();
            return;
        }
        if (this.xi[1] < CAP) {
            res.necrosis();
            return;
        }
        if (NPH - Math.log10(this.xi[3]) < HAP) {
            res.apoptosis();
            return;
        }
        if (this.xi[2] < GAP) {
            res.necrosis();
            return;
        }

        
        let f = Math.max(K * (max - TR) + 1, 0.25);
        let u = Cell.get_u(o, f);
        let p = true;
        let n_xi = Cell.add(this.xi, u);
        if (max == o[1] /*programmed quiescence*/ || n_xi[1] < 0 /*too little oxygen*/ || n_xi[2] < 0 /*too little glucose*/ || y[this.a][this.b].xi[0] > NAP /*contact inhibition*/) {
            f /= Q; // quiescence
            u = Cell.get_u(o, f);
            n_xi = Cell.add(this.xi, u);
            p = false;
        }

        if (n_xi[1] < 0 || n_xi[2] < 0) {
            res.necrosis();
            return;
        }
        
        res.xi = n_xi;
        res.age += f * DT;
        
        if (p && res.age >= res.split_age) {
            // proliferation
            res.age = 0;
            
            y[res.a][res.b] = res;
            
            let ok = [];
            for (let i = 0; i < DA.length; i++) {
                let na = res.a + DA[i], nb = res.b + DB[i];
                if (Cell.in_bounds(na, nb) && 
                   (y[na][nb].status == "empty"
                 || y[na][nb].status == "apoptosis")) {
                    ok.push([na, nb]);
                }
            }
            let rnd = Math.floor(Math.random() * ok.length);
            if (ok.length == 0) return;
            
            console.assert(ok.length);
            let [na, nb] = ok[rnd];
            let nres = Cell.mutate(res);
            nres.status = "alive";
            nres.xi = structuredClone(x[na][nb].xi);
            nres.a = na;
            nres.b = nb;
            y[na][nb] = nres;
            
            for (let i = 0; i < DA.length; i++) {
                let ma = na + DA[i], mb = nb + DB[i];
                if (Cell.in_bounds(ma, mb)) {
                    y[ma][mb].xi[0] = Cell.get_neighbors(ma, mb);
                }
            }
            y[na][nb].xi[0] = Cell.get_neighbors(na, nb);

            return;
        }

        // quiescence
        y[res.a][res.b] = res;

        // MOVEMENT FATE: Move the cell to a random empty neighbor slot
        if (Math.random() < AP) {
            let ok = [];
            for (let i = 0; i < DA.length; i++) {
                let na = this.a + DA[i], nb = this.b + DB[i];
                if (Cell.in_bounds(na, nb) && (
                    y[na][nb].status == "empty"
                 || y[na][nb].status == "apoptosis")) {
                    ok.push([na, nb]);
                }
            }
            if (ok.length > 0) {
                let [na, nb] = ok[Math.floor(Math.random() * ok.length)];
                res.a = na;
                res.b = nb;
                y[na][nb] = res; // Place cell in new location
                y[this.a][this.b] = new Cell(); // Clear old location
                y[this.a][this.b].set(x, y, this.a, this.b); // Reset to empty cell state
                
                // Recalculate neighborhood density scales
                y[na][nb].xi[0] = Cell.get_neighbors(na, nb);
                return;
            }
        }
    }
    
    // AI-Generated
    // Source: Gemini
    static diffuse_chemicals(x, y) {
        // Calculate diffusion coefficients (alpha = D * dt / dx^2)
        const alpha_c = (D_C * DT) / (DX * DX);
        const alpha_g = (D_G * DT) / (DX * DX);
        const alpha_h = (D_H * DT) / (DX * DX);

        // 1. Extract the Right Hand Side (RHS): the current concentrations 
        // after the consumption/production reactions applied in tick()
        let rhs_c = Array.from({length: N}, () => new Float64Array(M));
        let rhs_g = Array.from({length: N}, () => new Float64Array(M));
        let rhs_h = Array.from({length: N}, () => new Float64Array(M));

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < M; j++) {
                rhs_c[i][j] = y[i][j].xi[1];
                rhs_g[i][j] = y[i][j].xi[2];
                rhs_h[i][j] = y[i][j].xi[3];
            }
        }

        // 2. Gauss-Seidel iterative relaxation to solve the implicit scheme
        // 10-20 iterations is generally sufficient for convergence at these alpha levels.
        const ITERATIONS = 15; 

        for (let iter = 0; iter < ITERATIONS; iter++) {
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < M; j++) {
                    
                    // Dirichlet Boundary Conditions (from Section 2.2 / Image 1 & 3)
                    // The boundary represents surrounding blood vessels with constant concentrations (scaled to 1.0)
                    if (i === 0 || i === N - 1 || j === 0 || j === M - 1) {
                        y[i][j].xi[1] = 1.0; // Oxygen boundary
                        y[i][j].xi[2] = 1.0; // Glucose boundary
                        y[i][j].xi[3] = 1.0; // Hydrogen ion boundary
                        continue;
                    }

                    // Fetch neighboring values from the y array (using the latest iterative guesses)
                    let nc_up = y[i-1][j].xi[1], nc_dn = y[i+1][j].xi[1];
                    let nc_lt = y[i][j-1].xi[1], nc_rt = y[i][j+1].xi[1];

                    let ng_up = y[i-1][j].xi[2], ng_dn = y[i+1][j].xi[2];
                    let ng_lt = y[i][j-1].xi[2], ng_rt = y[i][j+1].xi[2];

                    let nh_up = y[i-1][j].xi[3], nh_dn = y[i+1][j].xi[3];
                    let nh_lt = y[i][j-1].xi[3], nh_rt = y[i][j+1].xi[3];

                    // Update concentrations using the solved implicit algebraic formula
                    // C_next = (C_rhs + alpha * (Sum of Neighbors)) / (1 + 4 * alpha)
                    y[i][j].xi[1] = (rhs_c[i][j] + alpha_c * (nc_up + nc_dn + nc_lt + nc_rt)) / (1.0 + 4.0 * alpha_c);
                    y[i][j].xi[2] = (rhs_g[i][j] + alpha_g * (ng_up + ng_dn + ng_lt + ng_rt)) / (1.0 + 4.0 * alpha_g);
                    y[i][j].xi[3] = (rhs_h[i][j] + alpha_h * (nh_up + nh_dn + nh_lt + nh_rt)) / (1.0 + 4.0 * alpha_h);
                }
            }
        }
    }
}

// AI-Generated
// Source: Gemini
function drawGrid() {
    ctxTumor.clearRect(0, 0, canvasTumor.width, canvasTumor.height);
    ctxOxygen.clearRect(0, 0, canvasOxygen.width, canvasOxygen.height);
    ctxGlucose.clearRect(0, 0, canvasGlucose.width, canvasGlucose.height);
    ctxAcid.clearRect(0, 0, canvasAcid.width, canvasAcid.height);

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            let cell = x[i][j];
            let cX = j * CELL_SIZE;
            let cY = i * CELL_SIZE;
            
            // 1. Tumor Context
            switch (cell.status) {
                case "alive": ctxTumor.fillStyle = "#2ecc71"; break;
                case "necrosis": ctxTumor.fillStyle = "#34495e"; break;
                case "apoptosis": ctxTumor.fillStyle = "#e74c3c"; break;
                case "empty": default: ctxTumor.fillStyle = "#f9f9f9"; break;
            }
            ctxTumor.fillRect(cX, cY, CELL_SIZE, CELL_SIZE);
            ctxTumor.strokeStyle = "#eee";
            ctxTumor.strokeRect(cX, cY, CELL_SIZE, CELL_SIZE);

            // 2. Oxygen Context
            let oxygen = Math.min(Math.max(cell.xi[1], 0), 1); 
            ctxOxygen.fillStyle = `rgb(0, ${Math.floor(oxygen * 120)}, ${Math.floor(oxygen * 255)})`;
            ctxOxygen.fillRect(cX, cY, CELL_SIZE, CELL_SIZE);
            ctxOxygen.strokeStyle = "#eee";
            ctxOxygen.strokeRect(cX, cY, CELL_SIZE, CELL_SIZE);

            // 3. Glucose Context
            let glucose = Math.min(Math.max(cell.xi[2], 0), 1);
            ctxGlucose.fillStyle = `rgb(${Math.floor(glucose * 230)}, ${Math.floor(glucose * 120)}, 0)`;
            ctxGlucose.fillRect(cX, cY, CELL_SIZE, CELL_SIZE);
            ctxGlucose.strokeStyle = "#eee";
            ctxGlucose.strokeRect(cX, cY, CELL_SIZE, CELL_SIZE);

            // 4. Proton/Acidity Context
            let acidity = Math.min(Math.max((cell.xi[3] - 1.0) / 1.2, 0), 1);
            ctxAcid.fillStyle = `rgb(255, ${Math.floor((1 - acidity) * 220)}, ${Math.floor((1 - acidity) * 220)})`;
            ctxAcid.fillRect(cX, cY, CELL_SIZE, CELL_SIZE);
            ctxAcid.strokeStyle = "#eee";
            ctxAcid.strokeRect(cX, cY, CELL_SIZE, CELL_SIZE);
        }
    }
}

let x = Array.from({length: N}, () => Array.from({length: M}, () => new Cell()));
let y = Array.from({length: N}, () => Array.from({length: M}, () => new Cell()));
for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
        x[i][j].set(x, y, i, j);
    }
}

x[N / 2][M / 2].status = "alive";

for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
        y[i][j].copy(x[i][j]);
    }
}
for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
        x[i][j].xi[0] = Cell.get_neighbors(i, j);
        y[i][j].copy(x[i][j]);
    }
}

function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}


function tick() {
    l = 0;
    r = 0;
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            y[i][j].copy(x[i][j]);
        }
    }

    let t = [];
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            t.push([i, j]);
        }
    }
    shuffle(t); 
    for (let [a, b] of t) {
        x[a][b].tick();
    }
    Cell.diffuse_chemicals(x, y);

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            x[i][j].copy(y[i][j]);
        }
    }

    // console.log(l / (l + r));

    drawGrid();

    requestAnimationFrame(tick);
}

tick();