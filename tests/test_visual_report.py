import os
import io
import time
import unittest
import numpy as np
import torch

from src.dataset import PML_apply_lowcut_fft
from src.visualization import (
    PML_save_examples_grid,
    PML_plot_mean_spectrum_compare,
)
from src.metrics import (
    PML_snr_lowband,
    PML_spectral_l2_bands,
)

F_DT = 0.004  # 4 ms, matches project default


def _two_tone_panel(H=512, W=16, f1=3.0, f2=24.0, amp2=0.8, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(H) * F_DT
    sig = np.sin(2*np.pi*f1*t)[:, None] + amp2*np.sin(2*np.pi*f2*t + rng.rand()*2*np.pi)[:, None]
    return np.repeat(sig, W, axis=1).astype(np.float32)


class TestVisualReport(unittest.TestCase):
    """
    Generates a browsable report with images and metrics so you can visually inspect the pipeline.
    The report is saved under:
        - env PML_REPORT_DIR, if set
        - otherwise: runs/vis_report/<timestamp>/
    """

    def test_generate_visual_report(self):
        # -------------------------------
        # Output directory
        # -------------------------------
        base_dir = os.environ.get("PML_REPORT_DIR", None)
        if base_dir is None or base_dir.strip() == "":
            base_dir = os.path.join("runs", "vis_report", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(base_dir, exist_ok=True)

        # -------------------------------
        # Synthetic data
        # -------------------------------
        H, W = 512, 16
        full = _two_tone_panel(H, W, f1=3.0, f2=26.0, amp2=0.8, seed=11)
        low  = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=10.0, norm="ortho")
        residual_true = full - low

        # Three recon candidates: bad, noisy, perfect
        rng = np.random.RandomState(7)
        recon_bad    = low.copy()
        recon_noisy  = (low + 0.8*residual_true + 0.05*rng.randn(*low.shape).astype(np.float32))
        recon_perfect = (low + residual_true)

        # -------------------------------
        # Save example grids (images & spectra per-sample)
        # -------------------------------
        x_batch = torch.from_numpy(np.stack([low,   low,   low  ], axis=0))[None, ...]   # (1,3,H,W)? -> fix to (3,1,H,W)
        y_batch = torch.from_numpy(np.stack([full,  full,  full ], axis=0))[None, ...]
        z_batch = torch.from_numpy(np.stack([recon_bad, recon_noisy, recon_perfect], axis=0))[None, ...]

        # Ensure shape (B,1,H,W)
        x_batch = x_batch.permute(1,0,2,3).contiguous()
        y_batch = y_batch.permute(1,0,2,3).contiguous()
        z_batch = z_batch.permute(1,0,2,3).contiguous()

        saved_paths = PML_save_examples_grid(
            x_batch, y_batch, z_batch, F_DT,
            c_dir=base_dir, n_max=3, c_prefix="sample", b_logy=False,
            l_bands=[(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)],
        )
        self.assertEqual(len(saved_paths), 3)
        for p in saved_paths:
            self.assertTrue(os.path.isfile(p))

        # -------------------------------
        # Comparison spectrum (all three on one plot)
        # -------------------------------
        fig_s, ax_s, outs = PML_plot_mean_spectrum_compare(
            [full, recon_bad, recon_noisy, recon_perfect],
            F_DT,
            l_labels=["full(target)", "recon_bad", "recon_noisy", "recon_perfect"],
            l_bands=[(0.0, 10.0), (10.0, 20.0)],
            b_logy=False, tn_figsize=(8, 4), c_title="Mean Spectra Comparison",
        )
        spec_path = os.path.join(base_dir, "spectra_compare.png")
        fig_s.savefig(spec_path, dpi=150)
        self.assertTrue(os.path.isfile(spec_path))

        # -------------------------------
        # Metrics (numeric checks + for report table)
        # -------------------------------
        torch_full  = torch.from_numpy(full)[None, None].float()
        torch_low   = torch.from_numpy(low)[None, None].float()
        torch_bad   = torch.from_numpy(recon_bad)[None, None].float()
        torch_noisy = torch.from_numpy(recon_noisy)[None, None].float()
        torch_perf  = torch.from_numpy(recon_perfect)[None, None].float()

        # SNR in low band should improve: bad < noisy < perfect
        snr_bad   = float(PML_snr_lowband(torch_bad,  torch_full, F_DT, f_fmax_low=10.0).item())
        snr_noisy = float(PML_snr_lowband(torch_noisy, torch_full, F_DT, f_fmax_low=10.0).item())
        snr_perf  = float(PML_snr_lowband(torch_perf,  torch_full, F_DT, f_fmax_low=10.0).item())

        # Value assertions (not just shapes)
        self.assertLess(snr_bad, snr_noisy + 1e-6)
        self.assertLess(snr_noisy, snr_perf + 1e-6)

        # Spectral L2 in low band should drop similarly
        bands = [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)]
        e_bad   = PML_spectral_l2_bands(torch_bad,  torch_full, F_DT, bands)[0]   # (3,)
        e_noisy = PML_spectral_l2_bands(torch_noisy, torch_full, F_DT, bands)[0]
        e_perf  = PML_spectral_l2_bands(torch_perf,  torch_full, F_DT, bands)[0]

        self.assertGreater(float(e_bad[0].item()), float(e_noisy[0].item()) - 1e-9)
        self.assertGreater(float(e_noisy[0].item()), float(e_perf[0].item()) - 1e-9)

        # -------------------------------
        # Build an HTML report
        # -------------------------------
        rows = []
        rows.append("<tr><th>Variant</th><th>SNR LF (0-10 Hz) [dB]</th>"
                    "<th>L2(0-10)</th><th>L2(10-20)</th><th>L2(20-40)</th></tr>")
        def fmt(name, snr, e):
            return (f"<tr><td>{name}</td>"
                    f"<td>{snr:.3f}</td>"
                    f"<td>{float(e[0].item()):.6e}</td>"
                    f"<td>{float(e[1].item()):.6e}</td>"
                    f"<td>{float(e[2].item()):.6e}</td></tr>")

        rows.append(fmt("recon_bad",   snr_bad,   e_bad))
        rows.append(fmt("recon_noisy", snr_noisy, e_noisy))
        rows.append(fmt("recon_perfect", snr_perf, e_perf))

        # Image block
        imgs_html = "".join(
            f'<div style="display:inline-block;margin:8px;text-align:center">'
            f'<img src="{os.path.basename(p)}" width="380"/><br>{os.path.basename(p)}'
            f'</div>'
            for p in saved_paths
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PipeML Visual Report</title>
<style>
body {{ font-family: Arial, sans-serif; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #888; padding: 6px 10px; }}
th {{ background: #eee; }}
</style>
</head>
<body>
<h1>PipeML Visual Report</h1>
<p>Directory: {base_dir}</p>

<h2>Metrics</h2>
<table>
{''.join(rows)}
</table>

<h2>Mean Spectra Comparison</h2>
<img src="{os.path.basename(spec_path)}" width="760"/>

<h2>Per-sample Grids</h2>
{imgs_html}

<p style="color:#777">Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>
"""
        html_path = os.path.join(base_dir, "index.html")
        with io.open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Sanity: report exists and nontrivial size
        self.assertTrue(os.path.isfile(html_path))
        self.assertGreater(os.path.getsize(html_path), 1024)

        # Optional print so it's easy to find when running manually
        print(f"\n[Visual report saved to] {html_path}\n")


if __name__ == "__main__":
    unittest.main()
