from logging import NullHandler, getLogger
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import unique_path

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def pdf_to_png(pdf_path: str, output_folder: str = None, png_name: str = None) -> None:
    """
    PDFファイルをPNG画像に変換する関数。

    各ページを個別のPNG画像として保存する。
    指定されたPNG名がある場合、同じ名前で保存するが、重複がある場合はユニークな名前に変更する。

    Parameters:
    - pdf_path (str): PDFファイルのパス。
    - output_folder (str, optional): 出力フォルダのパス。指定しない場合、PDFファイルと同じフォルダに保存される。
    - png_name (str, optional): 保存するPNGファイルの名前。指定しない場合、デフォルトの名前が使用される。

    Returns:
    - None
    """
    # PDFファイルを開く
    pdf_document = fitz.open(pdf_path)
    pdf_path = Path(pdf_path)

    # 出力フォルダを設定（指定がない場合はPDFファイルと同じ名前にする）
    if output_folder is None:
        output_folder = pdf_path.parent / pdf_path.stem
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 各ページをPNG画像として保存
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()

        # PNGファイル名の設定（指定がある場合はユニークな名前に変更する）
        if png_name:
            if not png_name.endswith(".png"):
                png_name += ".png"
            output_path = unique_path(output_folder / png_name)
        else:
            output_path = output_folder / f"{pdf_path.stem}_{page_num + 1}.png"

        # 画像を保存
        pix.save(output_path)
        logger.debug(f"Saved {output_path}")


def tiff_to_png(
    tiff_path: str, output_folder: str = None, png_name: str = None
) -> None:
    """
    マルチページのTIFFファイルをPNG画像に変換する関数。

    各ページを個別のPNG画像として保存する。

    Parameters:
    - tiff_path (str): TIFFファイルのパス。
    - output_folder (str, optional): 出力フォルダのパス。指定しない場合、TIFFファイルと同じフォルダに保存される。
    - png_name (str, optional): 保存するPNGファイルの名前。指定しない場合、デフォルトの名前が使用される。

    Returns:
    - None
    """
    # TIFFファイルを開く
    tiff_image = Image.open(tiff_path)
    num_pages = tiff_image.n_frames
    tiff_path = Path(tiff_path)

    # 出力フォルダを設定（指定がない場合はTIFFファイルと同じ名前にする）
    if output_folder is None:
        output_folder = tiff_path.parent / tiff_path.stem
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 各ページをPNG画像として保存
    for i in range(num_pages):
        tiff_image.seek(i)
        if png_name:
            if not png_name.endswith(".png"):
                png_name += ".png"
            output_path = unique_path(output_folder / png_name)
        else:
            output_path = output_folder / f"{tiff_path.stem}_{i + 1}.png"
        tiff_image.save(output_path, "PNG")
        logger.debug(f"Saved {output_path}")


def convert_to_png(
    input_path: str, output_folder: str = None, png_name: str = None
) -> None:
    """
    PDFまたはTIFFファイルをPNG画像に変換する関数。

    各ページを個別のPNG画像として保存する。
    指定されたPNG名がある場合、同じ名前で保存するが、重複がある場合はユニークな名前に変更する。

    Parameters:
    - input_path (str): 入力ファイルのパス（PDFまたはTIFF）。
    - output_folder (str, optional): 出力フォルダのパス。指定しない場合、入力ファイルと同じフォルダに保存される。
    - png_name (str, optional): 保存するPNGファイルの名前。指定しない場合、デフォルトの名前が使用される。

    Returns:
    - None
    """
    input_path = Path(input_path)
    if output_folder is None:
        output_folder = input_path.parent / input_path.stem
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".pdf":
        pdf_to_png(input_path, output_folder, png_name)
    elif input_path.suffix.lower() in [".tiff", ".tif"]:
        tiff_to_png(input_path, output_folder, png_name)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}")


def image_to_pdf(image_files: list[str], output_pdf: str) -> None:
    """
    画像ファイルのリストを単一のPDFファイルに変換する関数。

    Parameters:
    - image_files (list[str]): 画像ファイルのパスのリスト。
    - output_pdf (str): 出力PDFファイルのパス。

    Returns:
    - None
    """
    images = [Image.open(image) for image in image_files]
    images = [img.convert("RGB") for img in images]
    output_pdf = Path(output_pdf)
    images[0].save(output_pdf, save_all=True, append_images=images[1:])
    logger.debug(f"Saved {output_pdf}")


def tiff_to_pdf(tiff_path: str, output_pdf: str = None) -> None:
    """
    TIFFファイルをPDFに変換する関数。

    Parameters:
    - tiff_path (str): TIFFファイルのパス。
    - output_pdf (str, optional): 出力PDFファイルのパス。指定しない場合、TIFFファイルと同じ場所に保存される。

    Returns:
    - None
    """
    # TIFFファイルを開き、各ページをリストに格納
    tiff_image = Image.open(tiff_path)
    num_pages = tiff_image.n_frames
    image_files = []
    for i in range(num_pages):
        tiff_image.seek(i)
        image_files.append(tiff_image.copy())

    # 出力PDFファイルのパスを設定（指定がない場合）
    tiff_path = Path(tiff_path)
    if output_pdf is None:
        output_pdf = tiff_path.with_suffix(".pdf")

    # images = [img.convert("RGB") for img in images]
    images = image_files
    output_pdf = Path(output_pdf)
    images[0].save(output_pdf, save_all=True, append_images=images[1:])
    logger.debug(f"Saved {output_pdf}")


def pdf_to_tiff(pdf_path: str, output_tiff: str = None) -> None:
    """
    PDFファイルをTIFFに変換する関数。

    Parameters:
    - pdf_path (str): PDFファイルのパス。
    - output_tiff (str, optional): 出力TIFFファイルのパス。指定しない場合、PDFファイルと同じ場所に保存される。

    Returns:
    - None
    """
    # PDFファイルを開き、各ページをリストに格納
    pdf_document = fitz.open(pdf_path)
    image_files = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_files.append(image)

    # 出力TIFFファイルのパスを設定（指定がない場合）
    pdf_path = Path(pdf_path)
    if output_tiff is None:
        output_tiff = pdf_path.with_suffix(".tiff")

    # リスト内の画像をTIFFに変換
    image_files[0].save(
        output_tiff, save_all=True, append_images=image_files[1:], format="TIFF"
    )
    logger.debug(f"Saved {output_tiff}")


# 使用例
if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # PDFまたはTIFFをPNGに変換
    # input_path = 'path/to/your/file.pdf'  # または 'path/to/your/file.tiff'
    # output_folder = 'path/to/output/folder'
    # png_name = 'output_image_name'
    for i in range(2, 10):
        print(i)
        convert_to_png(rf"C:\組基\2024年\プレス\P-2024070{i}.pdf")

    # # 画像をPDFに変換
    # image_files = ['path/to/your/image1.png', 'path/to/your/image2.png']
    # output_pdf = 'path/to/output/file.pdf'
    # image_to_pdf(image_files, output_pdf)
    # pdf_to_tiff(
    #     r"C:\Users\yamamotok\Downloads\「令和6年度住民税額通知書」確認ガイド.pdf"
    # )
    # TIFFをPDFに変換
    # tiff_path = "path/to/your/file.tiff"
    # output_pdf = "path/to/output/file.pdf"
    # tiff_to_pdf(
    #     r"C:\Users\yamamotok\Downloads\a.tiff",
    # )

    # # PDFをTIFFに変換
    # pdf_path = 'path/to/your/file.pdf'
    # output_tiff = 'path/to/output/file.tiff'
    # pdf_to_tiff(pdf_path, output_tiff)


# if __name__ == "__main__":

# pdf_path = "path/to/your/file.pdf"
# tiff_path = "path/to/your/file.tif"
# png_files = ["path/to/your/image1.png", "path/to/your/image2.png"]
# output_folder = "path/to/output/folder"
# output_pdf = "path/to/output/file.pdf"


# for i in get_files(r"C:\Users\yamamotok\Documents\ss"):
#     pdf_to_png(i, r"C:\Users\yamamotok\Documents\ss\ab", "image")
# tiff_to_png(tiff_path, output_folder)
# image_to_pdf(png_files, output_pdf)
