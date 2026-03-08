import re
from dataclasses import dataclass
from typing import Iterable, List, Pattern, Sequence


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])\1+", r"\1", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"(,\s*)+,", ",", text)
    return text.strip()


def _compile_phrase_pattern(terms: Iterable[str]) -> Pattern[str]:
    normalized = sorted({t.strip().lower() for t in terms if t and t.strip()}, key=lambda x: (-len(x), x))
    escaped = []
    for term in normalized:
        token = re.escape(term)
        token = token.replace(r"\ ", r"\s+")
        escaped.append(token)

    if not escaped:
        return re.compile(r"$^")

    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class VisualVocabulary:
    colors: Sequence[str]
    materials: Sequence[str]
    textures: Sequence[str]
    insect_terms: Sequence[str]


DEFAULT_VOCAB = VisualVocabulary(
    colors=(
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "orange", "purple",
        "pink", "brown", "beige", "cyan", "magenta", "turquoise", "teal", "maroon", "navy", "violet",
        "indigo", "gold", "silver", "ivory", "cream", "olive", "tan", "peach", "mint", "burgundy",
        "crimson", "scarlet", "lavender", "lilac", "azure", "aqua", "aquamarine", "navy blue", "sky blue",
        "baby blue", "light blue", "dark blue", "light green", "dark green", "forest green", "lime green",
        "light red", "dark red", "rose red", "wine red", "light pink", "hot pink", "dark gray", "light gray",
        "dark grey", "light grey",
    ),
    materials=(
        "cotton", "wool", "silk", "linen", "denim", "leather", "suede", "velvet", "satin", "chiffon",
        "polyester", "nylon", "spandex", "acrylic", "rayon", "cashmere", "fleece", "corduroy", "lace",
        "mesh", "canvas", "tweed", "felt", "rubber", "plastic", "metal", "steel", "iron", "aluminum",
        "bronze", "brass", "ceramic", "glass", "wood", "bamboo", "stone", "marble", "granite",
        "concrete", "clay", "paper", "fur", "shearling", "down", "feather", "foam",
    ),
    textures=(
        "smooth", "rough", "soft", "hard", "glossy", "matte", "shiny", "dull", "coarse", "fine",
        "grainy", "fuzzy", "fluffy", "silky", "velvety", "wrinkled", "crumpled", "woven", "knit",
        "striped", "plaid", "checkered", "polka dot", "dotted", "spotted", "paisley", "floral",
        "camouflage", "camo", "animal print", "zebra print", "leopard print", "snake print", "herringbone",
        "chevron", "geometric", "abstract", "tie dye", "ombre", "gradient", "marbled", "transparent",
        "translucent", "opaque", "frosted", "sheer", "netted",
    ),
    insect_terms=(
        "black", "brown", "dark brown", "light brown", "tan", "beige", "cream", "white", "off white",
        "gray", "grey", "charcoal", "slate", "ash", "red", "reddish", "orange", "yellow", "green", "blue",
        "purple", "pink", "magenta", "violet", "rust", "russet", "chestnut", "mahogany", "ochre", "umber",
        "sienna", "tawny", "fawn", "amber", "golden", "bronze", "coppery", "pale", "dusky", "faded",
        "drab", "rufous", "testaceous", "fulvous", "ferruginous", "castaneous", "fuscous", "livid",
        "piceous", "violaceous", "cyaneous", "glaucous", "mottled", "blotchy", "flecked", "stained",
        "tinged", "tinted", "smudged", "clouded", "diffuse", "suffused", "melanized", "depigmented",
        "discolored", "frosted", "pruinose", "powdery", "mealy", "chalky", "dusty", "granular", "velvety",
        "shaded", "darkened", "lightened", "somber", "sooty", "smoky", "iridescent", "metallic",
        "submetallic", "opalescent", "pearlescent", "prismatic", "lustrous", "holographic", "glossy",
        "subglossy", "matte", "satiny", "silky", "polished", "reflective", "non reflective", "transparent",
        "translucent", "semi translucent", "opaque", "hyaline", "subhyaline", "smoky hyaline", "warm toned",
        "cool toned", "earthy", "vivid", "bright", "dark colored", "silken", "glassy", "resinous", "waxy",
        "glistening", "gleaming",
    ),
)


class VisualTermFilter:
    """Domain-aware caption filter for visual term removal."""

    def __init__(self, vocab: VisualVocabulary = DEFAULT_VOCAB, min_content_tokens: int = 2):
        self.min_content_tokens = max(1, int(min_content_tokens))
        self._color_pattern = _compile_phrase_pattern(vocab.colors)
        self._material_pattern = _compile_phrase_pattern(vocab.materials)
        self._texture_pattern = _compile_phrase_pattern(vocab.textures)
        self._insect_pattern = _compile_phrase_pattern(vocab.insect_terms)

    def filter_text(
        self,
        text: str,
        *,
        remove_insect: bool = False,
        remove_colors: bool = False,
        remove_materials: bool = False,
        remove_textures: bool = False,
    ) -> str:
        if not isinstance(text, str):
            return text

        original = _normalize_whitespace(text)
        if not original:
            return original

        cleaned = original
        if remove_insect:
            cleaned = self._insect_pattern.sub(" ", cleaned)
        if remove_colors:
            cleaned = self._color_pattern.sub(" ", cleaned)
        if remove_materials:
            cleaned = self._material_pattern.sub(" ", cleaned)
        if remove_textures:
            cleaned = self._texture_pattern.sub(" ", cleaned)

        cleaned = _normalize_whitespace(cleaned)

        content_tokens = [
            tok for tok in re.findall(r"\b\w+\b", cleaned.lower()) if tok not in STOPWORDS
        ]
        if len(content_tokens) < self.min_content_tokens:
            return original

        return cleaned

    def filter_batch(self, texts: Sequence[str], **kwargs) -> List[str]:
        return [self.filter_text(t, **kwargs) for t in texts]
