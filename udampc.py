import numpy as np
import scipy.linalg as la

class UDAMPC:
    def __init__(self, mpc):
        self.mpc = mpc   # pass your MPCCont object here

    def predictor(self, A, B, x, x_ref, d=1, prev_us=None):
            # 1. Compute A^d
        A_d = np.linalg.matrix_power(A, d)

        # 2. Build the summation term sum(A^i B u_{k-1-i})
        nx, nu = B.shape
        sum_term = np.zeros((nx,))
        if prev_us is not None and len(prev_us) > 0:
            for i in range(min(d, len(prev_us))):
                sum_term += np.linalg.matrix_power(A, i) @ B @ prev_us[i]

        # 3. Predict future state after delay
        x_pred = A_d @ x + sum_term

        # 4. Update model with predicted state and solve MPC
        self.mpc.update_model(A, B, x_pred, x_ref)
        u = self.mpc.solve()

        return u

    def augmented(self, A, B, x, x_ref, d=2):
        """Augmented-state delay handling."""
        nx, nu = A.shape[0], B.shape[1]

        # Augmented system (basic shift register structure)
        A_aug = np.block([
            [A, B, np.zeros((nx, (d-1)*nu))],
            [np.zeros((nu*(d-1), nx)), np.eye(nu*(d-1)), np.zeros((nu*(d-1), nu))],
            [np.zeros((nu, nx+nu*(d-1))), np.zeros((nu, nu))]
        ])
        B_aug = np.vstack([np.zeros((nx, nu)), np.zeros(((d-1)*nu, nu)), np.eye(nu)])

        x_aug = np.concatenate([x, np.zeros(d*nu)])
        x_ref_aug = np.concatenate([x_ref, np.zeros(d*nu)])

        self.mpc.update_model(A_aug, B_aug, x_aug, x_ref_aug)
        u_aug = self.mpc.solve()
        return u_aug[:nu]
    
        
    def compute_rpi_ellipsoid(self, A, B, K, w_bound, Q=None):
        """
        Compute ellipsoidal invariant set for error dynamics e_{k+1}=(A-BK)e_k + w_k.
        Returns shape matrix P and radius rho.
        """
        import scipy.linalg as la
        n = A.shape[0]
        Ak = A - B @ K
        if Q is None:
            Q = np.eye(n)

        # Solve discrete Lyapunov: Ak^T P Ak - P = -Q
        P = la.solve_discrete_lyapunov(Ak.T, Q)

        # Ellipsoid bound radius (conservative, scaled)
        rho = (w_bound**2) * np.trace(P)
        return P, rho


    def tube(self, A, B, x, x_ref, K, w_bound=0.05):
        """
        Tube MPC with ellipsoidal tightening (anisotropic, safe version).
        """
        # Step 1: compute invariant ellipsoid
        P, rho = self.compute_rpi_ellipsoid(A, B, K, w_bound)

        # Step 2: project ellipsoid into input space via K
        P_inv = np.linalg.inv(P)
        margins = []
        for i in range(B.shape[1]):   # each input channel
            k_i = K[i,:].reshape(-1,1)
            margin_i = np.sqrt(max(rho, 0)) * np.sqrt(float(k_i.T @ P_inv @ k_i))

            # CAP margin so it never exceeds 20% of actuator authority
            if i < 3:   # forces
                margin_cap = 0.1 * np.min(np.abs([self.mpc.u_min[i], self.mpc.u_max[i]]))
            else:       # torques
                margin_cap = 0.3 * np.min(np.abs([self.mpc.u_min[i], self.mpc.u_max[i]]))

            margin_i = min(margin_i, margin_cap)
            margins.append(margin_i)

        u_margin = np.array(margins)
        print("Ellipsoidal margins (capped):", u_margin)

        # Step 3: tighten input constraints anisotropically
        u_min_orig, u_max_orig = self.mpc.u_min.copy(), self.mpc.u_max.copy()
        self.mpc.u_min = u_min_orig + u_margin
        self.mpc.u_max = u_max_orig - u_margin

        # Step 4: solve nominal MPC
        self.mpc.update_model(A, B, x, x_ref)
        u_nom = self.mpc.solve()

        # Step 5: restore original bounds
        self.mpc.u_min, self.mpc.u_max = u_min_orig, u_max_orig

        # Step 6: add feedback correction (scaled)
        u_fb = -0.12 * (K @ (x - x_ref))   # tune 0.05–0.2 range
        u_total = u_nom + u_fb
        u_total = np.clip(u_total, u_min_orig, u_max_orig)
        return u_total



    # def compute_rpi_ellipsoid(self, Ak, W):
    #     """
    #     Compute P solving: Ak P Ak^T - P + W = 0
    #     """
    #     import numpy as np
    #     import scipy.linalg as la
    #     try:
    #         P = la.solve_discrete_lyapunov(Ak, W)
    #     except Exception as e:
    #         eigs = np.linalg.eigvals(Ak)
    #         if np.any(np.abs(eigs) >= 1.0):
    #             Ak_reg = 0.99 * Ak
    #             P = la.solve_discrete_lyapunov(Ak_reg, W)
    #         else:
    #             raise
    #     P = 0.5*(P + P.T) + 1e-12*np.eye(P.shape[0])
    #     return P


    # def tube_control(self, A, B, x, x_ref, K, w_bound=0.05,
    #                  cap_frac_force=0.2, cap_frac_torque=0.3):
    #     import numpy as np
    #     n = A.shape[0]
    #     W = (w_bound**2) * np.eye(n)
    #     P = self.compute_rpi_ellipsoid(A - B @ K, W)   

    #     margins = []
    #     for i in range(B.shape[1]):
    #         k_i = K[i, :].reshape(1, -1)
    #         margin_i = float(np.sqrt(k_i @ P @ k_i.T))
    #         margins.append(margin_i)
    #     u_margin = np.array(margins)

    #     u_min_orig = self.mpc.u_min.copy()
    #     u_max_orig = self.mpc.u_max.copy()
    #     act_range = 0.5*(np.abs(u_min_orig) + np.abs(u_max_orig))
    #     caps = np.zeros_like(u_margin)
    #     caps[:3] = cap_frac_force * act_range[:3]
    #     caps[3:] = cap_frac_torque * act_range[3:]
    #     u_margin = np.minimum(u_margin, caps)

    #     self.mpc.u_min = u_min_orig + u_margin
    #     self.mpc.u_max = u_max_orig - u_margin

    #     self.mpc.update_model(A, B, x, x_ref)
    #     u_nom = self.mpc.solve().flatten()

    #     self.mpc.u_min, self.mpc.u_max = u_min_orig, u_max_orig

    #     fb_scale = 0.04
    #     u_fb = -fb_scale * (K @ (x - x_ref))
    #     u_total = u_nom + u_fb
    #     u_total = np.clip(u_total, u_min_orig, u_max_orig)
    #     return u_total




