"""Generate QR code posters pointing to the live demo site."""

import os
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask
from PIL import Image, ImageDraw, ImageFont

URL = "https://presentationsite-six.vercel.app"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def try_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_qr(path: str, url: str, fg=(5, 7, 11), bg=(255, 255, 255), rounded=False, size_px=1200):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 30% redundancy
        box_size=10,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)

    if rounded:
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=RoundedModuleDrawer(radius_ratio=0.9),
            color_mask=SolidFillColorMask(back_color=bg, front_color=fg),
        ).convert("RGB")
    else:
        img = qr.make_image(fill_color=fg, back_color=bg).convert("RGB")

    # Upscale to target pixel size
    if img.size[0] < size_px:
        img = img.resize((size_px, size_px), Image.NEAREST)
    img.save(path)
    print(f"[QR]  {path}   ({img.size[0]}x{img.size[1]}, url={url})")
    return img


def make_poster(out_path: str, qr_img: Image.Image, url: str):
    """Print-ready A-sized poster with heading + QR + footer."""
    W, H = 1240, 1748  # A4 @ 150 DPI
    poster = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(poster)

    # Fonts — fall back gracefully
    f_title = try_font(["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"], 72)
    f_sub = try_font(["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"], 42)
    f_body = try_font(["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"], 34)
    f_mono = try_font(["consolab.ttf", "Consolas.ttf", "DejaVuSansMono-Bold.ttf"], 32)
    f_small = try_font(["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"], 26)

    accent = (8, 145, 178)  # cyan
    dark = (15, 23, 42)
    dim = (100, 116, 139)

    # Top bar
    draw.rectangle([0, 0, W, 14], fill=accent)

    # Title block
    y = 90
    draw.text((80, y), "Real site vs. model prediction", font=f_title, fill=dark)
    y += 105
    draw.text((80, y), "Interactive 3D demo — scan to open",
              font=f_sub, fill=dim)
    y += 75

    # Divider
    draw.line([(80, y), (W - 80, y)], fill=(226, 232, 240), width=2)
    y += 50

    # Body copy
    body_lines = [
        "XGBoost trained on 2,043 proteins with 225 structural,",
        "dynamic, pocket, and ESM-2 features per residue.",
        "Blind-tested on 2,370 independent CASBench proteins.",
        "",
        "Tap any of the 5 demo proteins. Switch between ground",
        "truth, prediction, and probability heatmap modes.",
        "Try 1HQ6 and 3W8L to see the novel pocket-bias failure.",
    ]
    for line in body_lines:
        draw.text((80, y), line, font=f_body, fill=dark)
        y += 48

    # QR code centered
    qr_size = 780
    qr_resized = qr_img.resize((qr_size, qr_size), Image.LANCZOS)
    qr_x = (W - qr_size) // 2
    qr_y = y + 30
    # Shadow / border box
    pad = 20
    draw.rectangle(
        [qr_x - pad, qr_y - pad, qr_x + qr_size + pad, qr_y + qr_size + pad],
        outline=(226, 232, 240),
        width=3,
    )
    poster.paste(qr_resized, (qr_x, qr_y))
    y = qr_y + qr_size + 60

    # URL
    url_text = url.replace("https://", "")
    tw = draw.textlength(url_text, font=f_mono)
    draw.text(((W - tw) // 2, y), url_text, font=f_mono, fill=accent)
    y += 60

    # Footer
    footer = "Hassan AL Habeeb  ·  Supervised by Dr. Mohammed Al Mohaini"
    tw = draw.textlength(footer, font=f_small)
    draw.text(((W - tw) // 2, H - 80), footer, font=f_small, fill=dim)

    poster.save(out_path, "PNG")
    print(f"[POSTER]  {out_path}   ({W}x{H})")


def main():
    # Plain QR — drop-in use anywhere
    qr_plain = make_qr(os.path.join(OUT_DIR, "qr_code.png"), URL, rounded=False)
    # Rounded-module QR — looks softer on slides
    make_qr(os.path.join(OUT_DIR, "qr_code_rounded.png"), URL, rounded=True)
    # Full printable A4 poster with heading
    make_poster(os.path.join(OUT_DIR, "qr_poster.png"), qr_plain, URL)
    print(f"\nEncoded URL: {URL}")


if __name__ == "__main__":
    main()
