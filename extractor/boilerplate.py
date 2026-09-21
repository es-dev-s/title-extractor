"""
Boilerplate / masthead denylist for title candidates.

This is the highest-leverage accuracy lever: grow it when a real PDF
fails. Patterns are applied to a whitespace-normalized line.
"""

from __future__ import annotations

import re
import unicodedata

# Exact (case-insensitive) lines that are never titles.
EXACT_REJECT = {
    "abstract",
    "introduction",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "appendices",
    "methodology",
    "discussion",
    "results",
    "related work",
    "background",
    "keywords",
    "key words",
    "contents",
    "table of contents",
    "list of figures",
    "list of figure",
    "list of tables",
    "list of table",
    "list of abbreviations",
    "index",
    "on",
    "by",
    "to",
    "a",
    "i",
    "ii",
    "iii",
    "iv",
    "nomenclature",
    "notation",
    "preface",
    "foreword",
    "summary",
    "synopsis",
    "disclosure",
    "funding",
    "open access",
    "paper open access",
    "you may also like",
    "to cite this article",
    "view the article online",
    "view the article online for updates and enhancements",
    "research article",
    "review article",
    "original article",
    "original research",
    "regular paper",
    "research paper",
    "case study",
    "short communication",
    "technical note",
    "editorial",
    "guest editorial",
    "correspondence",
    "letter to the editor",
    "full length article",
    "available online",
    "accepted manuscript",
    "in press",
    "preprint",
    "not peer-reviewed",
    "peer reviewed",
    "copyright",
    "all rights reserved",
    "published by",
    "downloaded from",
    "creative commons",
    "conflict of interest",
    "author contributions",
    "data availability",
    "supplementary material",
    "supporting information",
    "highlights",
    "graphical abstract",
    "article info",
    "article history",
    "manuscript information",
    "research article",
    "easychair preprint",
    "manuscript info",
    "project report",
    "minor project report",
    "a project report submitted",
    "a case study report",
    "submitted by",
    "submitted to",
    "guided by",
    "open in app",
    "sign in",
    "sign up",
    "design steps",
    "design example",
    "learning resource center",
    "general guidelines",
    "aims and scope",
    "table of contents",
    "editorial board",
    "internal editor",
    "editor in chief",
    "bachelor of technology",
    "bachelor of engineering",
    "engineering and management",
    "of engineering",
    "of engineering management",
    "college of engineering",
    "institute of engineering",
    "bonafide certificate",
    "certificate",
    "archive of sid ir",
    "archive of sid.ir",
    "scientific african",
    "irjet",
    "open",
    "open access",
    "master s degree thesis",
    "master's degree thesis",
    "research article",
}

# Generic PDF / Word processor leftovers — used for /Title metadata too.
GENERIC_TITLES = {
    "untitled",
    "untitled document",
    "document",
    "document1",
    "new document",
    "microsoft word",
    "microsoft word document",
    "presentation",
    "worksheet",
    "workbook",
    "layout 1",
    "layout1",
    "scan",
    "scanned",
    "unknown",
    "null",
    "none",
    "title",
    "paper",
    "draft",
    "final",
    "final draft",
    "report",
    "pdf",
    "export",
    "output",
    "file",
    "slide 1",
    "slide1",
    "cn report",
}

_FLAGS = re.IGNORECASE | re.UNICODE

# (pattern, reason) — a match is a hard reject for layout candidates.
LINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bissn\b", _FLAGS), "issn"),
    (re.compile(r"\bisbn\b", _FLAGS), "isbn"),
    (re.compile(r"\bdoi\s*:?\s*10\.\d", _FLAGS), "doi"),
    (re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", _FLAGS), "doi"),
    (re.compile(r"\barxiv\s*:?\s*\d", _FLAGS), "arxiv"),
    (re.compile(r"https?://", _FLAGS), "url"),
    (re.compile(r"\bwww\.", _FLAGS), "url"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", _FLAGS), "email"),
    (re.compile(r"\borcid\b", _FLAGS), "orcid"),
    (re.compile(r"available\s+online", _FLAGS), "available_online"),
    (re.compile(r"\bvol(?:ume)?\.?\s*\d+", _FLAGS), "volume"),
    (re.compile(r"\bno(?:\.|umber)?\s*\d+", _FLAGS), "issue"),
    (re.compile(r"\bissue\s*\d+", _FLAGS), "issue"),
    (re.compile(r"\bpp?\.?\s*\d+\s*[–-]\s*\d+", _FLAGS), "pages"),
    (re.compile(r"©|copyright\s+\d{4}", _FLAGS), "copyright"),
    (re.compile(r"all\s+rights\s+reserved", _FLAGS), "copyright"),
    (re.compile(r"\breceived\b.{0,40}\baccepted\b", _FLAGS), "dates"),
    (re.compile(r"^(received|accepted|revised|published|available)\s*[:.]?", _FLAGS), "dates"),
    (re.compile(r"date of submission|submitted date", _FLAGS), "dates"),
    (re.compile(r"corresponding\s+author", _FLAGS), "corresponding_author"),
    (re.compile(r"^(keywords|key\s*words)\s*[:.]", _FLAGS), "keywords"),
    (re.compile(r"^(abstract|summary|highlights)\s*[:.\u2014\-]", _FLAGS), "abstract_label"),
    (re.compile(r"furnished to the author for internal non-commercial", _FLAGS), "elsevier_cover"),
    (re.compile(r"this article appeared in a journal published by", _FLAGS), "elsevier_cover"),
    (re.compile(r"author.?s personal copy", _FLAGS), "elsevier_cover"),
    (re.compile(r"sharing with colleagues", _FLAGS), "elsevier_cover"),
    (re.compile(r"selling or licensing copies", _FLAGS), "elsevier_cover"),
    (re.compile(r"posting to personal, institutional or third party", _FLAGS), "elsevier_cover"),
    (re.compile(r"other uses, including reproduction and distribution", _FLAGS), "elsevier_cover"),
    (re.compile(r"messages\.(studocu|downloaded_by|pdf_cover)", _FLAGS), "studocu"),
    (re.compile(r"lomoarcpsd", _FLAGS), "studocu"),
    (re.compile(r"not sponsored or endorsed", _FLAGS), "studocu"),
    (re.compile(r"archive of sid", _FLAGS), "watermark"),
    (re.compile(r"easychair preprint", _FLAGS), "preprint_banner"),
    (re.compile(r"see discussions, stats, and author profiles", _FLAGS), "researchgate"),
    (re.compile(r"openinapp|open in app", _FLAGS), "browser_chrome"),
    (re.compile(r"https?://medium\.com", _FLAGS), "medium_chrome"),
    (re.compile(r"\bmin\s*read\b", _FLAGS), "medium_chrome"),
    (re.compile(r"\|\s*by\s+.+\bMedium\b", _FLAGS), "medium_chrome"),
    (re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}.+\bMedium\b", _FLAGS), "medium_chrome"),
    (re.compile(r"^a\s+project\s+report\b", _FLAGS), "front_matter"),
    (re.compile(r"^(course|section|group|semester)\s*[:\-]\s*.{0,48}$", _FLAGS), "course_label"),
    (re.compile(r"^(?:college\s+)?of\s+engineering\s*(?:&|and)\s*management$", _FLAGS), "institution"),
    (re.compile(r"^ud[cs]\s+", _FLAGS), "classification"),
    (re.compile(r"\bdergisi\b|\bzeitschrift\b", _FLAGS), "journal_foreign"),
    (re.compile(r"research journal$", _FLAGS), "journal"),
    (re.compile(r"all right reserved", _FLAGS), "copyright"),
    (re.compile(r"a journal produced by", _FLAGS), "journal_cover"),
    (re.compile(r"^(figure|fig\.?|table|eq(?:uation)?)\s*\d", _FLAGS), "caption"),
    (re.compile(r"contents lists available", _FLAGS), "sciencedirect"),
    (re.compile(r"\birjet\b", _FLAGS), "journal"),
    (re.compile(r"^scientific african$", _FLAGS), "journal"),
    (re.compile(r"journal homepage", _FLAGS), "journal"),
    (re.compile(r"creative\s+commons|cc[\s-]?by", _FLAGS), "license"),
    (re.compile(r"this\s+is\s+an\s+open\s+access", _FLAGS), "open_access"),
    (re.compile(r"you\s+may\s+also\s+like", _FLAGS), "iop_chrome"),
    (re.compile(r"paper\s*[•·\-]?\s*open\s+access", _FLAGS), "iop_chrome"),
    (re.compile(r"to\s+cite\s+this\s+article", _FLAGS), "iop_chrome"),
    (re.compile(r"view\s+the\s+article\s+online", _FLAGS), "iop_chrome"),
    (re.compile(r"this\s+content\s+was\s+downloaded\s+from", _FLAGS), "iop_chrome"),
    (re.compile(r"\biop\s+(?:publishing|conference series)\b", _FLAGS), "publisher"),
    (re.compile(r"downloaded\s+from", _FLAGS), "download_banner"),
    (re.compile(r"licensed\s+under", _FLAGS), "license"),
    (re.compile(r"\belsevier\b|\bspringer(?:\s+nature)?\b|\bwiley\b|\bsage\s+publications\b", _FLAGS), "publisher"),
    (re.compile(r"\bmdpi\b|\btaylor\s*&\s*francis\b|\binderscience\b|\biaeme\b", _FLAGS), "publisher"),
    (re.compile(r"peer[\s-]?review", _FLAGS), "peer_review"),
    (re.compile(r"manuscript\s+(id|number|received)", _FLAGS), "manuscript"),
    (re.compile(r"article\s+(in\s+press|history|info)", _FLAGS), "article_info"),
    (re.compile(r"\bsupplementary\s+(material|data|information)\b", _FLAGS), "supplement"),
]

MASTHEAD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjournal\s+of\b", _FLAGS), "journal"),
    (re.compile(r"\binternational\s+journal\b", _FLAGS), "journal"),
    (re.compile(r"\bproceedings\s+of\b", _FLAGS), "proceedings"),
    (re.compile(r"\btransactions\s+(on|of)\b", _FLAGS), "transactions"),
    (re.compile(r"\bconference\s+on\b", _FLAGS), "conference"),
    (re.compile(r"\bsymposium\s+on\b", _FLAGS), "symposium"),
    (re.compile(r"\bieee\b|\bacm\b|\basme\b|\bsae\b|\bsiam\b", _FLAGS), "society"),
    (re.compile(r"\bjournal\s+homepage\b", _FLAGS), "journal"),
    (re.compile(r"\bresearch journal\b", _FLAGS), "journal"),
    (re.compile(r"\bconference series\b", _FLAGS), "journal"),
    (re.compile(r"^iop\s+conference", _FLAGS), "journal"),
    (re.compile(r"^scientific reports$", _FLAGS), "journal"),
]

AFFILIATION_RE = re.compile(
    r"\b(university|universit[eé]|universität|department|dept\.|institute|"
    r"college|faculty|laboratory|laboratoire|school of|center for|centre for|"
    r"research center|research centre|gmbh|pvt\.?\s*ltd|inc\.|campus)\b",
    _FLAGS,
)

AUTHOR_NAME_LIST_RE = re.compile(
    r"^[^\W\d_][\w.'’-]+(?:\s+[^\W\d_][\w.'’.-]+){0,4}"
    r"(?:\s*,\s*[^\W\d_][\w.'’-]+(?:\s+[^\W\d_][\w.'’.-]+){0,4}){1,}$",
    _FLAGS,
)

INSTITUTION_LINE_RE = re.compile(
    r"^(?:the\s+)?(?:national\s+|international\s+)?"
    r"(?:university|institute|college|department|faculty|school)\b",
    _FLAGS,
)

LABELED_TITLE_RE = re.compile(
    r"^(?:project\s+titt?le|thesis\s+title|report\s+title|title)\s*[:\-]\s*(.+)$",
    _FLAGS,
)

# Medium / print-to-PDF chrome: "8/30/26, 1:25 PM | Article Title | by Author | Medium"
BROWSER_CHROME_TITLE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}[^|]*\|\s*(.+?)\s*\|\s*by\s+",
    _FLAGS,
)

COURSE_LINE_RE = re.compile(r"^(?:course|subject)\s*[:\-]\s*(.+)$", _FLAGS)
COURSE_PREFIX_RE = re.compile(
    r"^(?:electrical\s+machines?|computer\s+networks|electronics|"
    r"mechanical(?:\s+engineering)?|civil(?:\s+engineering)?)(?:\s+|$)",
    _FLAGS,
)

COURSE_CODE_RE = re.compile(r"^[A-Z]{2,}\s*[-]?\s*\d{3,}\b")

COVER_PAGE_RE = re.compile(
    r"furnished to the author for internal non-commercial|"
    r"this article appeared in a journal published by|"
    r"sharing with colleagues|"
    r"selling or licensing copies|"
    r"posting to personal, institutional or third party|"
    r"messages\.studocu|lomoarcpsd|learning resource center|"
    r"general guidelines",
    _FLAGS,
)

SECTION_NUMBER_RE = re.compile(
    r"^(?:[\divxl]+)(?:\.\d+)*[.)]?\s+",
    _FLAGS,
)

FILE_EXTENSION_RE = re.compile(
    r"\.(pdf|docx?|odt|rtf|txt|pptx?|xlsx?|html?|tex|zip)\b",
    _FLAGS,
)

WORD_PROCESSOR_RE = re.compile(
    r"microsoft\s+word|libreoffice|openoffice|google\s+docs|adobe\s+acrobat|"
    r"powerpoint|word\s*-\s*|pages\s+document",
    _FLAGS,
)

PATH_RE = re.compile(r"^[a-z]:\\|\\\\|/users/|/home/|/tmp/", _FLAGS)


def normalize_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


def exact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(text).lower()).strip()
