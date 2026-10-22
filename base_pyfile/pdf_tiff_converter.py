"""PDF / TIFF / 画像同士の変換ユーティリティ。

PyMuPDF (fitz) と Pillow を使う。重い依存なので、パッケージ本体からは
遅延 import される（``base_pyfile.pdf_tiff_converter`` を直接 import したときだけ読み込まれる）。
"""

from contextlib import ExitStack
from logging import NullHandler, getLogger
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
from PIL import Image

from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import unique_path

# Image.new で作った画像を PDF として保存する場合などに備え、
# JPEG などの標準プラグインを明示的に初期化しておく。
Image.init()

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

ImagePath = Union[str, Path]


def _png_output_path(
    output_folder: Path,
    png_name: str | None,
    default_stem: str,
    page_num: int,
) -> str:
    """PNG の出力パスを決める。既存ファイルがある場合は連番を付ける。"""
    if png_name:
        name = png_name if png_name.lower().endswith(".png") else f"{png_name}.png"
        return unique_path(output_folder / name)
    return unique_path(output_folder / f"{default_stem}_{page_num + 1}.png")


def pdf_to_png(
    pdf_path: ImagePath,
    output_folder: ImagePath | None = None,
    png_name: str | None = None,
    dpi: int | None = None,
) -> None:
    """PDFファイルをPNG画像に変換する関数。

    各ページを個別のPNG画像として保存する。既存ファイルがある場合は
    上書きせず、``name_1.png`` のような連番を付ける。

    Args:
        pdf_path (str | Path): PDFファイルのパス。
        output_folder (str | Path, optional): 出力フォルダ。省略時はPDFと同じ階層の
            PDF名フォルダに保存する。
        png_name (str, optional): 保存するPNGファイル名。省略時は
            ``PDF名_ページ番号.png`` を使用する。
        dpi (int, optional): 出力解像度。省略時は PyMuPDF の既定値（72dpi）を使う。

    Raises:
        ValueError: ``dpi`` が 1 未満の場合。
    """
    if dpi is not None and dpi <= 0:
        raise ValueError("dpi は 1 以上で指定してください。")

    pdf_path = Path(pdf_path)
    output_folder = (
        Path(output_folder)
        if output_folder is not None
        else pdf_path.parent / pdf_path.stem
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72 if dpi is not None else None

    # with を使い、Windows でもPDFファイルを開いたままにしない。
    with fitz.open(str(pdf_path)) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            if zoom is None:
                pix = page.get_pixmap()
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            output_path = _png_output_path(
                output_folder, png_name, pdf_path.stem, page_num
            )
            pix.save(str(output_path))
            logger.debug(f"Saved {output_path}")


def tiff_to_png(
    tiff_path: ImagePath,
    output_folder: ImagePath | None = None,
    png_name: str | None = None,
) -> None:
    """マルチページTIFFファイルをPNG画像に変換する関数。

    各ページを個別のPNG画像として保存する。既存ファイルがある場合は
    上書きせず、連番を付ける。

    Args:
        tiff_path (str | Path): TIFFファイルのパス。
        output_folder (str | Path, optional): 出力フォルダ。省略時はTIFFと同じ階層の
            TIFF名フォルダに保存する。
        png_name (str, optional): 保存するPNGファイル名。省略時は
            ``TIFF名_ページ番号.png`` を使用する。
    """
    tiff_path = Path(tiff_path)
    output_folder = (
        Path(output_folder)
        if output_folder is not None
        else tiff_path.parent / tiff_path.stem
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    with Image.open(tiff_path) as tiff_image:
        for page_num in range(tiff_image.n_frames):
            tiff_image.seek(page_num)
            output_path = _png_output_path(
                output_folder, png_name, tiff_path.stem, page_num
            )
            tiff_image.save(str(output_path), "PNG")
            logger.debug(f"Saved {output_path}")


def convert_to_png(
    input_path: ImagePath,
    output_folder: ImagePath | None = None,
    png_name: str | None = None,
    dpi: int | None = None,
) -> None:
    """PDFまたはTIFFファイルをPNG画像に変換する関数。

    Args:
        input_path (str | Path): 入力ファイル（PDFまたはTIFF）。
        output_folder (str | Path, optional): 出力フォルダ。
        png_name (str, optional): 保存するPNGファイル名。
        dpi (int, optional): PDFの出力解像度。TIFFでは無視される。

    Raises:
        ValueError: 対応していない拡張子、または ``dpi`` が 1 未満の場合。
    """
    input_path = Path(input_path)
    output_folder = (
        Path(output_folder)
        if output_folder is not None
        else input_path.parent / input_path.stem
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        pdf_to_png(input_path, output_folder, png_name, dpi=dpi)
    elif suffix in {".tiff", ".tif"}:
        tiff_to_png(input_path, output_folder, png_name)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def image_to_pdf(image_files: List[ImagePath], output_pdf: ImagePath) -> None:
    """画像ファイルのリストを1つのPDFファイルに変換する関数。

    Args:
        image_files (list[str | Path]): 画像ファイルのパスのリスト。
        output_pdf (str | Path): 出力PDFファイルのパス。

    Raises:
        ValueError: 画像リストが空の場合。
    """
    if not image_files:
        raise ValueError("image_files が空です。PDFを作成できません。")

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # 自分で開いた画像だけを閉じる。呼び出し元から渡された PIL.Image は閉じない。
    with ExitStack() as stack:
        rgb_images: List[Image.Image] = []
        for image in image_files:
            if isinstance(image, (str, Path)):
                source = stack.enter_context(Image.open(image))
            elif isinstance(image, Image.Image):
                source = image
            else:
                raise TypeError(f"画像パスまたは PIL.Image を指定してください: {image!r}")
            converted = source.convert("RGB")
            if converted is not source:
                stack.callback(converted.close)
            rgb_images.append(converted)

        rgb_images[0].save(
            output_pdf, save_all=True, append_images=rgb_images[1:]
        )
    logger.debug(f"Saved {output_pdf}")


def tiff_to_pdf(tiff_path: ImagePath, output_pdf: ImagePath | None = None) -> None:
    """TIFFファイルをPDFに変換する関数。

    Args:
        tiff_path (str | Path): TIFFファイルのパス。
        output_pdf (str | Path, optional): 出力PDFファイルのパス。省略時は
            TIFFと同じ場所に同じ名前で保存する。

    Raises:
        ValueError: TIFFにページがない場合。
    """
    tiff_path = Path(tiff_path)
    output_pdf = (
        Path(output_pdf) if output_pdf is not None else tiff_path.with_suffix(".pdf")
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    rgb_images: List[Image.Image] = []
    with Image.open(tiff_path) as tiff_image:
        for page_num in range(tiff_image.n_frames):
            tiff_image.seek(page_num)
            rgb_images.append(tiff_image.convert("RGB"))

        if not rgb_images:
            raise ValueError("TIFFにページがありません。PDFを作成できません。")

        try:
            rgb_images[0].save(
                output_pdf, save_all=True, append_images=rgb_images[1:]
            )
        finally:
            for image in rgb_images:
                image.close()
    logger.debug(f"Saved {output_pdf}")


def pdf_to_tiff(pdf_path: ImagePath, output_tiff: ImagePath | None = None) -> None:
    """PDFファイルをTIFFに変換する関数。

    Args:
        pdf_path (str | Path): PDFファイルのパス。
        output_tiff (str | Path, optional): 出力TIFFファイルのパス。省略時は
            PDFと同じ場所に同じ名前で保存する。

    Raises:
        ValueError: PDFにページがない場合。
    """
    pdf_path = Path(pdf_path)
    output_tiff = (
        Path(output_tiff) if output_tiff is not None else pdf_path.with_suffix(".tiff")
    )
    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    image_files: List[Image.Image] = []
    with fitz.open(str(pdf_path)) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            image_files.append(
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            )

        if not image_files:
            raise ValueError("PDFにページがありません。TIFFを作成できません。")

        try:
            image_files[0].save(
                output_tiff,
                save_all=True,
                append_images=image_files[1:],
                format="TIFF",
            )
        finally:
            for image in image_files:
                image.close()
    logger.debug(f"Saved {output_tiff}")


# 使用例
if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # 実際のパスに置き換えて実行する
    # PDF または TIFF を PNG に変換
    # convert_to_png("path/to/your/file.pdf", "path/to/output/folder")

    # 画像を PDF に変換
    # image_to_pdf(["path/to/image1.png", "path/to/image2.png"], "path/to/output/file.pdf")

    # PDF を TIFF に変換
    # pdf_to_tiff("path/to/your/file.pdf", "path/to/output/file.tiff")

    # TIFF を PDF に変換
    # tiff_to_pdf("path/to/your/file.tiff", "path/to/output/file.pdf")
    pass
