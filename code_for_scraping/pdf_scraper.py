import re
import PyPDF2

input_pdf_pth = r'.\LOTR_PageFormatted.pdf'
def parse_page_indices(page_str):
    """
    Parse a string like '1,2,5,10-12' into a sorted list of unique 0-based page indices.
    """
    indices = set()
    for part in page_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            indices.update(range(int(start)-1, int(end)))
        elif part.isdigit():
            indices.add(int(part)-1)
    return sorted(indices)

def extract_pdf_pages(input_pdf_path, output_pdf_path, page_indices_str):
    """
    Extracts specified pages from a PDF and saves them to a new PDF.
    Args:
        input_pdf_path: Path to the input PDF file.
        output_pdf_path: Path to save the new PDF file.
        page_indices_str: String like '1,2,5,10-12' (1-based page numbers).
    """
    reader = PyPDF2.PdfReader(input_pdf_path)
    writer = PyPDF2.PdfWriter()
    indices = parse_page_indices(page_indices_str)
    for idx in indices:
        if 0 <= idx < len(reader.pages):
            writer.add_page(reader.pages[idx])
    with open(output_pdf_path, 'wb') as f:
        writer.write(f)
    print(f"Saved extracted pages to {output_pdf_path}")

gandalf_pages = '9,11,13,24-27,32-36,39-69,74,75,76,83,84,87,103,105-109,132-133,140,141,151,166-174,183,186-189,191,195,198,199,206-210,219-232,238,239,247-272,275-336,355-368,393,396,397,402,414,416,435,437,441,444,453,465-467,472,492-529,539-560,563,567,570,572-600,615,640,644,647,670-671,677,679,682,685-686,687,693,700,708,732,747-761,765-768,772,773,776,780,793,796,799,806-829,843,850-859,862-873,877-885,888-892,934,947-956,966,968,970-971,975,979-980,983-996,1018,1030,1055-1056,1060,1069,1077-1078,1080,1085-1093,1096,1133'

sam_pages = '13,14,22,24,44-47,50,58,61-214,219,220,223-227,230,231,233,237,238,239,271-407,413,414,419,439,472,482,490,496,571,590,603-742,792,796,811-812,815,889,897-947,950-957,966,970,975,982-1031,1044,1090-1099,1103,1105,1107,1112,1138'

frodo_pages = '2,10,13,15,21-22,23,28,30-407,413,414,415,418,419,426,434,440,444,450,453,482,490,495-496,516,571,603-742,748,749,753,792,796,808,811-812,813,815,879,880,885,887,889,890,892,897,898,899,902,903,904,907,908,910-957,966,968,970,972,974-975,982,984-1030,1041,1078,1090-1096,1100-1104,1112,1133'

legolas_pages = '240,255,272,275,279-405,414-443,488-550,556-577,585,586,658,773,775-792,796,848,872-878,883,886,888,954-957,970,976,978,981,1080,1081,1098'

extract_pdf_pages(
    input_pdf_pth,
    'output.pdf',
    legolas_pages
)

