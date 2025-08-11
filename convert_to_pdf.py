#!/usr/bin/env python3
"""
Convert Technical Report Markdown to PDF
"""

import markdown
import os
from pathlib import Path

def markdown_to_html(markdown_file, output_file):
    """Convert markdown to HTML with styling"""
    
    # Read the markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['codehilite', 'tables', 'toc'])
    html_content = md.convert(markdown_content)
    
    # Create a complete HTML document with CSS styling
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reinforcement Learning Tutorial System - Technical Report</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 24px;
            margin-bottom: 16px;
        }}
        h1 {{
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
            font-size: 28px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 8px;
            font-size: 24px;
        }}
        h3 {{
            font-size: 20px;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 90%;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 85%;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 16px;
            color: #7f8c8d;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 16px;
            margin: 20px 0;
        }}
        .toc h2 {{
            margin-top: 0;
            border-bottom: none;
        }}
        @page {{
            margin: 1in;
            size: A4;
        }}
        @media print {{
            body {{
                max-width: none;
                margin: 0;
                padding: 0;
            }}
            h1, h2, h3, h4, h5, h6 {{
                page-break-after: avoid;
            }}
            pre, blockquote {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML file created: {output_file}")

def html_to_pdf_weasyprint(html_file, pdf_file):
    """Convert HTML to PDF using weasyprint"""
    try:
        from weasyprint import HTML, CSS
        HTML(filename=html_file).write_pdf(pdf_file)
        print(f"PDF created successfully: {pdf_file}")
        return True
    except ImportError:
        print("WeasyPrint not available. HTML file created instead.")
        return False
    except Exception as e:
        print(f"Error creating PDF with WeasyPrint: {e}")
        return False

def html_to_pdf_alternative(html_file, pdf_file):
    """Alternative PDF conversion using pdfkit"""
    try:
        import pdfkit
        options = {
            'page-size': 'A4',
            'margin-top': '1in',
            'margin-right': '1in',
            'margin-bottom': '1in',
            'margin-left': '1in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_file, pdf_file, options=options)
        print(f"PDF created successfully with pdfkit: {pdf_file}")
        return True
    except ImportError:
        print("pdfkit not available.")
        return False
    except Exception as e:
        print(f"Error creating PDF with pdfkit: {e}")
        return False

def main():
    """Main conversion function"""
    
    # File paths
    docs_dir = Path("docs")
    markdown_file = docs_dir / "TECHNICAL_REPORT_ORIGINAL.md"
    html_file = docs_dir / "TECHNICAL_REPORT_ORIGINAL.html"
    pdf_file = docs_dir / "TECHNICAL_REPORT_ORIGINAL.pdf"
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"Error: Markdown file not found: {markdown_file}")
        return
    
    print(f"Converting {markdown_file} to PDF...")
    
    # Step 1: Convert markdown to HTML
    markdown_to_html(markdown_file, html_file)
    
    # Step 2: Convert HTML to PDF
    success = False
    
    # Try WeasyPrint first
    print("Attempting PDF conversion with WeasyPrint...")
    success = html_to_pdf_weasyprint(html_file, pdf_file)
    
    # If WeasyPrint fails, try pdfkit
    if not success:
        print("Attempting PDF conversion with pdfkit...")
        success = html_to_pdf_alternative(html_file, pdf_file)
    
    if success:
        print(f"\n✅ PDF conversion completed successfully!")
        print(f"📄 PDF file: {pdf_file}")
        print(f"🌐 HTML file: {html_file}")
    else:
        print(f"\n⚠️ PDF conversion failed, but HTML file is available:")
        print(f"🌐 HTML file: {html_file}")
        print("\nYou can:")
        print("1. Open the HTML file in a web browser and print to PDF")
        print("2. Install wkhtmltopdf and try again")
        print("3. Use an online HTML to PDF converter")

if __name__ == "__main__":
    main()
