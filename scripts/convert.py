# This script uses the `markdownify` library to convert html content to Markdown format.
# in order to use this script you need to run `pip install markdownify` first.
# The following code will read and store the HTML content from a file of your choice.
html = open("./testing/scripts/file.html", "r").read()
# Output location for the converted Markdown content
output_path = "./testing/scripts/content.md"

# The code is largely based on the `markdownify` library but extends it to handle specific elements
# as following the instructions on the github page for the library (https://github.com/matthewwithanm/python-markdownify),
# this script extends the `MarkdownConverter` class, and overrides some of the methods with custom logic.
#
# Some of the methods used module definitions of functions that were not able to be overridden inside the class.
# As changes to these methods are impossible to do without modifying the library, this script redefines them,
# modifies them, and then redefines the two functions "process_tag" and "process_text" to use the new definitions.
#
# Modifications are covered under the MIT license, as the original library is under the MIT license. This allows for
# free use and modification of the code.
#
# TLDR; The script extends markdownify's main class and `monkey-patches` some elements to handle specific formatting

# CODE:

# Import the necessary libraries:

# This is the base library for converting HTML to Markdown
from markdownify import markdownify as md
# Functions required to monkey-patch & the MarkdownConverter class
from markdownify import MarkdownConverter, re_all_whitespace, re_newline_whitespace, re_whitespace
from markdownify import re_html_heading, re_extract_newlines
#HTML parsing tools
from bs4 import Comment, Doctype, NavigableString, Tag
# Utilities used inside of markdownify to run older python code
import six

# This following are slightly modified versions of the functions
# that are used inside of the MarkdownConverter class.
def should_remove_whitespace_inside(el):
    """Return to remove whitespace immediately inside a block-level element."""
    if not el or not el.name:
        return False
    # Start modification
    if el.name == 'p' and el.parent.name == 'div' and el.parent.parent.name == 'div':
        # If this is a <p> inside a <div> inside another <div>, treat it as a preformatted block.
        return False  # Do not remove whitespace inside <pre> blocks.
    # End modification
    if re_html_heading.match(el.name) is not None:
        return True
    return el.name in ('p', 'blockquote',
                       'article', 'div', 'section',
                       'ol', 'ul', 'li',
                       'dl', 'dt', 'dd',
                       'table', 'thead', 'tbody', 'tfoot',
                       'tr', 'td', 'th')


def should_remove_whitespace_outside(el): # No modification, necessary to include.
    """Return to remove whitespace immediately outside a block-level element."""
    return should_remove_whitespace_inside(el) or (el and el.name == 'pre')

# This is the main class that extends the MarkdownConverter class
# and overrides some of the methods to handle specific elements.
class XrdConfigConverter(MarkdownConverter):
    """Converts XrootD config to Markdown. Extends the MarkdownConverter class to handle specific elements."""

    def convert_div(self, el, text, parent_tags): # This logic is used to handel the code sections inside of the documentation.
        if (el.parent.name == "div"):
            markdowned = super().convert_div(el, text, parent_tags)
            return '\n'.join(f'> {line}' if line.strip() else '>' for line in markdowned.splitlines())
        return super().convert_div(el, text, parent_tags)
    def convert_p(self, el, text, parent_tags): # This logic is used to handle the code sections inside of the documentation.
        if (el.parent.name == "div" and el.parent.parent.name == "div"):
            parent_tags.add('pre') # This treats the <p> inside a <div> inside another <div> as a preformatted block.
            return super().convert_p(el, text, parent_tags)
        return super().convert_p(el, text, parent_tags)
    
    # These methods have slight modifications to handle the specific elements. This code has original comments and logic from the original library. 
    def process_tag(self, node, parent_tags=None):
        # For the top-level element, initialize the parent context with an empty set.
        if parent_tags is None:
            parent_tags = set()

        # Collect child elements to process, ignoring whitespace-only text elements
        # adjacent to the inner/outer boundaries of block elements.
        should_remove_inside = should_remove_whitespace_inside(node)

        # Start main modification --------------------------------------------------------------------------
        if node.name == 'p' and node.parent.name == 'div' and node.parent.parent.name == 'div':
            # If this is a <p> inside a <div> inside another <div>, treat it as a preformatted block.
            parent_tags.add('pre')
            parent_tags.add('_nostrip')
            should_remove_inside = False  # Do not remove whitespace inside <pre> blocks.
        # End main modification --------------------------------------------------------------------------

        def _can_ignore(el):
            if isinstance(el, Tag):
                # Tags are always processed.
                return False
            elif isinstance(el, (Comment, Doctype)):
                # Comment and Doctype elements are always ignored.
                # (subclasses of NavigableString, must test first)
                return True
            elif isinstance(el, NavigableString):
                if six.text_type(el).strip() != '':
                    # Non-whitespace text nodes are always processed.
                    return False
                elif should_remove_inside and (not el.previous_sibling or not el.next_sibling):
                    # Inside block elements (excluding <pre>), ignore adjacent whitespace elements.
                    return True
                elif should_remove_whitespace_outside(el.previous_sibling) or should_remove_whitespace_outside(el.next_sibling):
                    # Outside block elements (including <pre>), ignore adjacent whitespace elements.
                    return True
                else:
                    return False
            elif el is None:
                return True
            else:
                raise ValueError('Unexpected element type: %s' % type(el))

        children_to_convert = [el for el in node.children if not _can_ignore(el)]

        # Create a copy of this tag's parent context, then update it to include this tag
        # to propagate down into the children.
        parent_tags_for_children = set(parent_tags)
        parent_tags_for_children.add(node.name)

        # if this tag is a heading or table cell, add an '_inline' parent pseudo-tag
        if (
            re_html_heading.match(node.name) is not None  # headings
            or node.name in {'td', 'th'}  # table cells
        ):
            parent_tags_for_children.add('_inline')

        # if this tag is a preformatted element, add a '_noformat' parent pseudo-tag
        if node.name in {'pre', 'code', 'kbd', 'samp'}:
            parent_tags_for_children.add('_noformat')

        # Convert the children elements into a list of result strings.
        child_strings = [
            self.process_element(el, parent_tags=parent_tags_for_children)
            for el in children_to_convert
        ]

        # Remove empty string values.
        child_strings = [s for s in child_strings if s]

        # Collapse newlines at child element boundaries, if needed.
        if node.name == 'pre' or node.find_parent('pre'):
            # Inside <pre> blocks, do not collapse newlines.
            pass
        else:
            # Collapse newlines at child element boundaries.
            updated_child_strings = ['']  # so the first lookback works
            for child_string in child_strings:
                # Separate the leading/trailing newlines from the content.
                leading_nl, content, trailing_nl = re_extract_newlines.match(child_string).groups()

                # If the last child had trailing newlines and this child has leading newlines,
                # use the larger newline count, limited to 2.
                if updated_child_strings[-1] and leading_nl:
                    prev_trailing_nl = updated_child_strings.pop()  # will be replaced by the collapsed value
                    num_newlines = min(2, max(len(prev_trailing_nl), len(leading_nl)))
                    leading_nl = '\n' * num_newlines

                # Add the results to the updated child string list.
                updated_child_strings.extend([leading_nl, content, trailing_nl])

            child_strings = updated_child_strings

        # Join all child text strings into a single string.
        text = ''.join(child_strings)

        # apply this tag's final conversion function
        convert_fn = self.get_conv_fn_cached(node.name)
        if convert_fn is not None:
            text = convert_fn(node, text, parent_tags=parent_tags)

        return text
    # Same explanation as previous method
    def process_text(self, el, parent_tags=None):
        # For the top-level element, initialize the parent context with an empty set.
        if parent_tags is None:
            parent_tags = set()

        text = six.text_type(el) or ''

        # normalize whitespace if we're not inside a preformatted element
        if 'pre' not in parent_tags:
            if self.options['wrap']:
                text = re_all_whitespace.sub(' ', text)
            else:
                text = re_newline_whitespace.sub('\n', text)
                text = re_whitespace.sub(' ', text)

        # escape special characters if we're not inside a preformatted or code element
        if '_noformat' not in parent_tags:
            text = self.escape(text, parent_tags)

        # remove leading whitespace at the start or just after a
        # block-level element; remove traliing whitespace at the end
        # or just before a block-level element.

        # Start Modification --------------------------------------------------------------------------
        if (should_remove_whitespace_outside(el.previous_sibling)
                or (should_remove_whitespace_inside(el.parent)
                    and not el.previous_sibling)) and '_nostrip' not in parent_tags: # Line modified
            text = text.lstrip(' \t\r\n')
        if (should_remove_whitespace_outside(el.next_sibling)
                or (should_remove_whitespace_inside(el.parent)
                    and not el.next_sibling)) and '_nostrip' not in parent_tags: # Line modified
            text = text.rstrip()
        
        # End Modification --------------------------------------------------------------------------
        return text

# Initialize the converter and convert the HTML content to Markdown
content = XrdConfigConverter(extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code'], strip=['a']).convert(html)

# Save the converted content to a markdown file path defined above
with open(output_path, "w", encoding="utf-8") as f:
    f.write(content)


