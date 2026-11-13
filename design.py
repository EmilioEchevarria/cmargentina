from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from colorspacious import cspace_convert
from scipy.interpolate import PchipInterpolator

UCS = "CAM02-UCS"

# Recipes for the colormaps, run `python design.py` to check the maps
# Needs colorspacious and scipy

def ramp(colors, n=256):
    """Smooth path through the anchor colours, resampled to equal perceptual steps."""
    jab = cspace_convert([to_rgb(c) for c in colors], "sRGB1", UCS)
    path = PchipInterpolator(np.linspace(0, 1, len(colors)), jab)(np.linspace(0, 1, 2000))
    rgb = np.clip(cspace_convert(path, UCS, "sRGB1"), 0, 1)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cspace_convert(rgb, "sRGB1", UCS), axis=0), axis=1))]
    return np.array([np.interp(np.linspace(0, s[-1], n), s, c) for c in rgb.T]).T


def bands(pairs, line, n=40):
    """Light-to-dark bands with a thin dark line between them."""
    out = [ramp(p, n) for p in pairs]
    for a, b in zip(out[:-1], out[1:]):
        a[-1] = b[0] = to_rgb(line)
    return np.vstack(out)


MAPS = {
    "sudestada": ramp("#073050 #124e96 #2079c0 #2fa5d8 #3fd1d9 #83f2ba #f4ee78 "
                      "#f8c562 #f79639 #ee5d2b #c5293d #8f1744 #590d40".split()),
    "fileteado": bands([("#b4cbf9", "#2043a3"), ("#b1e9f7", "#008399"), ("#bff4c9", "#0e8c41"),
                        ("#fff4ca", "#c5a500"), ("#ffd9bb", "#c86f00"), ("#ffb5ab", "#bb0a10"),
                        ("#e6a0ba", "#760044")], line="#181928"),
    "caminito": [to_rgb(c) for c in "#2a61b1 #179765 #6a499b #e87f25 #68b5e2 #c52c26 #ddad0f #c65b93".split()],
    "pampa": ramp("#fef7dd #fed990 #f7b83d #b3a90f #279c47 #007654 #005148 #002d2f".split()),
    "ferro": ramp("#f3f9ef #bbdcb4 #72b875 #1a8443 #004128".split()),
    "malbec": ramp("#fff4f3 #f2babf #d56c85 #912d58 #3b122e".split()),
    "asado": ramp("#0f0705 #571712 #b1341f #ed8121 #fec25e #fdf4d2".split()),
    "sol_de_mayo": np.vstack([ramp("#301f31 #4264a5 #7cb2e2 #eeeeee".split(), 128),
                              ramp("#eeeeee #dda016 #9c4d1e #301f31".split(), 128)]),
    "mate": np.vstack([ramp("#071e40 #185a89 #4d9ebc #96d0d7".split(), 128),
                       ramp("#34622a #5d7e28 #99933a #c7a570 #e0c6a8 #edebe7".split(), 128)]),
}

CODE = '''
for _name, _rgb in MAPS.items():
    _cmap = ListedColormap(np.array(_rgb.split(), float).reshape(-1, 3), _name)
    for _c in (_cmap, _cmap.reversed()):
        mpl.colormaps.register(_c, force=True)
        globals()[_c.name] = _c


def show():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(MAPS), figsize=(7, 0.45 * len(MAPS)))
    for ax, name in zip(axs, MAPS):
        ax.imshow([globals()[name].colors], aspect="auto")
        ax.text(-0.02, 0.5, name, transform=ax.transAxes, ha="right", va="center")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.2)
    plt.show()
'''

if __name__ == "__main__":
    data = "".join(f'    "{k}": """\n' + "\n".join(" ".join(f"{x:.4f}" for x in c) for c in v) + '\n""",\n' for k, v in MAPS.items())
    Path(__file__).with_name("cmargentina.py").write_text(
        "import numpy as np\nimport matplotlib as mpl\nfrom matplotlib.colors import ListedColormap\n\n"
        "MAPS = {\n" + data + "}\n" + CODE)

    deut = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}
    fig, ax = plt.subplots(len(MAPS), 2, figsize=(11, 1.6 * len(MAPS)))
    for (name, rgb), (a, b) in zip(MAPS.items(), ax):
        rgb = np.array(rgb)
        jab = cspace_convert(rgb, "sRGB1", UCS)
        gray = cspace_convert(jab * [1, 0, 0], UCS, "sRGB1")
        a.imshow(np.clip([rgb, cspace_convert(rgb, deut, "sRGB1"), gray], 0, 1), aspect="auto")
        a.set(title=name, yticks=[0, 1, 2], yticklabels=["normal", "deuteranopia", "grayscale"], xticks=[])
        b.plot(jab[:, 0], label="lightness")
        step = np.linalg.norm(np.diff(jab, axis=0), axis=1)
        b.plot(50 * step / step.mean(), label="step size (flat = even)")
        b.legend(fontsize=7)
    fig.tight_layout()
    plt.show()
