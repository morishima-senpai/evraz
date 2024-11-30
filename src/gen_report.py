import markdown2
import pdfkit
import pygments
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

class MarkdownToPDFConverter:
    def __init__(self, output_file=None, theme='default'):
        """
        Initialize the Markdown to PDF converter
        
        :param output_file: Path for the output PDF (automatically adds `.pdf` at the end)
        :param theme: CSS theme for styling (default/github/custom)
        """
        self.output_file = output_file + '.pdf'
        self.theme = theme
        
    def _get_css_theme(self):
        """
        Get CSS theme for PDF rendering
        
        :return: CSS content for theming
        """
        themes = {
            'default': """
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px; 
                }
                code { 
                    background-color: #f4f4f4; 
                    padding: 2px 4px; 
                    border-radius: 4px; 
                }
                pre { 
                    background-color: #f4f4f4; 
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto; 
                    white-space: pre-wrap; /* Ensures text wraps */
                    word-wrap: break-word; /* Breaks long words */
                }
            """,
            'github': """
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 30px;
                    color: #24292e;
                }
                pre {
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    font-size: 85%;
                    line-height: 1.45;
                    overflow: auto;
                    padding: 16px;
                }
            """
        }
        return themes.get(self.theme, themes['default'])
    
    def _syntax_highlight(self, markdown_text : str) -> str:
        """
        Add syntax highlighting to code blocks
        
        :param markdown_text: Markdown text to process
        :return: Markdown text with syntax highlighting
        """
        def highlight_code(match):
            code = match.group(2)
            lang = match.group(1) or 'text'
            try:
                lexer = get_lexer_by_name(lang)
                formatter = HtmlFormatter()
                highlighted_code = pygments.highlight(code, lexer, formatter)
                return f'<pre><code class="highlighted">{highlighted_code}</code></pre>'
            except Exception:
                return match.group(0)
        
        import re
        return re.sub(r'```(\w+)?\n(.*?)```', highlight_code, markdown_text, flags=re.DOTALL)
    
    def convert(self, markdown_text : str):
        """
        Convert Markdown to PDF
        
        :return: Path to generated PDF
        """
        
        # Syntax highlighting
        markdown_text = self._syntax_highlight(markdown_text)
        
        # Convert Markdown to HTML
        html = markdown2.markdown(markdown_text, extras=['tables', 'fenced-code-blocks'])
        
        # Create full HTML with CSS
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                {self._get_css_theme()}
                {HtmlFormatter().get_style_defs('.highlighted')}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        pdfkit.from_string(full_html, self.output_file, 
                           options={
                               'page-size': 'A4',
                               'margin-top': '0.75in',
                               'margin-right': '0.75in',
                               'margin-bottom': '0.75in',
                               'margin-left': '0.75in',
                           })
        
        # return self.output_file


