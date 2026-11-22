"Closed word lists used by the deterministic caption parser (parse_facts.py)."

COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black",
    "white", "gray", "grey", "silver", "gold", "golden", "beige", "tan", "maroon",
    "navy", "teal", "turquoise", "violet", "cream", "ivory", "colorful", "colourful",
    "multicolored", "multicoloured", "blond", "blonde", "crimson", "magenta", "khaki",
    "olive", "lavender", "burgundy", "copper-colored",
}

SHADES = {"dark", "light", "bright", "pale", "deep", "vivid"}
MATERIALS = {
    "wooden", "wood", "metal", "metallic", "steel", "iron", "brick", "stone", "glass",
    "plastic", "leather", "concrete", "marble", "ceramic", "paper", "cardboard",
    "cloth", "fabric", "cotton", "wool", "woolen", "denim", "rubber", "straw", "wicker",
    "bamboo", "granite", "tile", "tiled", "silk", "velvet", "porcelain", "aluminum",
    "aluminium", "copper", "brass", "chrome", "stainless", "furry", "fur",
}
SIZES = {
    "small", "large", "big", "tall", "short", "tiny", "huge", "little", "long", "wide",
    "narrow", "giant", "massive", "enormous", "thick", "thin", "high", "low",
    "miniature", "oversized", "medium", "medium-sized", "small-sized", "full-sized",
}
SHAPES = {"round", "square", "rectangular", "circular", "oval", "triangular", "curved",
          "flat", "pointed", "spherical", "cylindrical", "straight"}
PATTERNS = {"striped", "checkered", "plaid", "spotted", "dotted", "floral", "patterned"}
SUBJECTIVE = {
    "beautiful", "lovely", "lively", "tranquil", "peaceful", "serene", "pleasant",
    "nice", "cute", "adorable", "happy", "joyful", "cheerful", "calm", "relaxing",
    "relaxed", "cozy", "cosy", "charming", "picturesque", "stunning", "gorgeous",
    "delightful", "fun", "enjoyable", "vibrant", "inviting", "friendly", "playful",
    "elegant", "impressive", "interesting", "unique", "perfect", "great", "wonderful",
    "scenic", "idyllic", "festive", "romantic", "dramatic", "energetic", "joyous",
    "content", "excited", "curious", "focused", "attentive", "casual", "leisurely",
    "warm", "welcoming", "pleasing", "appealing", "delicious", "tasty", "appetizing",
    "fresh", "busy", "bustling", "lovely", "fascinating", "captivating", "intriguing",
}
WEATHER = {"sunny", "cloudy", "rainy", "snowy", "foggy", "misty", "overcast", "stormy",
           "windy", "hazy", "clear", "sunlit", "snow-covered"}
TIME = {"night", "nighttime", "daytime", "evening", "morning", "afternoon", "sunset",
        "sunrise", "dusk", "dawn", "twilight", "midday", "noon", "day"}
INDOOR = {"indoor": "indoor", "indoors": "indoor", "outdoor": "outdoor",
          "outdoors": "outdoor", "inside": "indoor", "outside": "outdoor"}
LIGHTING = {"dimly", "dim", "brightly", "lit", "dark", "sunlit", "well-lit", "shadowy"}
PLACES = {
    "park", "beach", "street", "road", "kitchen", "bathroom", "bedroom", "room",
    "office", "market", "city", "town", "village", "restaurant", "cafe", "café",
    "store", "shop", "airport", "station", "stadium", "field", "meadow", "pasture",
    "forest", "woods", "garden", "yard", "backyard", "farm", "zoo", "classroom",
    "library", "highway", "intersection", "sidewalk", "river", "lake", "ocean", "sea",
    "mountain", "mountains", "desert", "countryside", "hallway", "lobby", "mall",
    "harbor", "harbour", "port", "plaza", "square", "courtyard", "playground",
    "campus", "church", "museum", "gym", "court", "arena", "studio", "warehouse",
    "factory", "hospital", "school", "parking", "lot", "runway", "track", "pier",
    "dock", "marina", "trail", "path", "alley", "neighborhood", "downtown", "suburb",
    "living", "dining", "bar", "pub", "bakery", "supermarket", "skatepark", "slope",
    "hill", "valley", "jungle", "savanna", "grassland", "lawn", "shore", "coast",
}
STYLE_NOUNS = {"photograph", "photo", "painting", "drawing", "sketch", "illustration",
               "cartoon", "render", "rendering", "screenshot", "poster", "collage",
               "watercolor", "artwork", "snapshot", "selfie", "portrait", "close-up",
               "closeup", "graphic"}
STYLE_ADJ = {"monochrome", "grayscale", "greyscale", "sepia", "vintage", "blurry",
             "blurred", "aerial", "panoramic", "cartoonish", "animated", "digital",
             "realistic", "abstract", "black-and-white", "close-up", "low-resolution",
             "high-resolution", "overhead", "candid", "artistic"}
META_NOUNS = {"image", "picture", "photo", "photograph", "scene", "view", "shot",
              "frame", "snapshot", "scenery", "composition", "setting"}
META_VERBS = {"feature", "show", "depict", "display", "capture", "contain", "include",
              "portray", "present", "showcase", "illustrate", "highlight", "showcase",
              "consist", "comprise", "focus"}

STATE_VERBS = {"see", "locate", "position", "place", "situate", "find", "notice",
               "observe", "scatter", "arrange", "set", "spread", "visible", "surround",
               "line", "park", "stack", "pile", "hang", "mount", "attach", "fill",
               "cover", "display"}
POSITION_NOUNS = {"front", "top", "left", "right", "side", "middle", "center", "centre",
                  "back", "bottom", "corner", "edge", "rear", "foreground", "background",
                  "distance", "end", "part", "area", "midst", "surface", "inside",
                  "outside", "underside", "base", "foot", "head", "half", "portion"}
ABSTRACT = {"moment", "time", "way", "atmosphere", "mood", "environment", "sense",
            "feeling", "activity", "life", "fun", "experience", "look", "appearance",
            "variety", "kind", "type", "sort", "amount", "total", "attention",
            "ambiance", "ambience", "vibe", "presence", "interest", "day", "scene",
            "setting", "event", "occasion", "game", "action", "motion", "style",
            "design", "color", "colour", "shape", "size", "number", "lot", "plenty",
            "location", "place", "spot", "view", "detail", "details", "element",
            "elements", "aspect", "touch", "contrast", "focus", "purpose",
            "possibility", "relationship", "connection", "bond", "camaraderie",
            "enjoyment", "leisure", "recreation", "excitement", "joy", "beauty",
            "nature", "weather", "addition", "middle", "variety", "array"}
QUANT_NOUNS = {"group", "couple", "pair", "bunch", "number", "lot", "lots", "crowd",
               "herd", "flock", "row", "stack", "pile", "set", "collection", "variety",
               "handful", "total", "dozen", "cluster", "line", "fleet", "pack", "bundle",
               "assortment", "array", "series", "plenty", "few", "range", "selection",
               "team", "family", "swarm", "bouquet", "batch"}
QUANT_DETS = {"several", "many", "few", "multiple", "numerous", "various", "both",
              "some", "countless", "plenty"}
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "dozen": 12, "pair": 2,
    "couple": 2, "both": 2, "single": 1,
}
COUNT_QUALIFIERS = {"at least", "at most", "about", "around", "approximately", "nearly",
                    "almost", "over", "more than", "less than", "fewer than", "roughly",
                    "up to", "exactly", "only", "just"}
HEDGE_ADV = {"possibly", "probably", "likely", "perhaps", "maybe", "apparently",
             "presumably", "seemingly", "potentially", "supposedly", "allegedly",
             "reportedly", "conceivably", "arguably", "plausibly"}
HEDGE_RAISING = {"appear", "seem", "look"}
HEDGE_INFER = {"suggest", "indicate", "imply", "hint", "signal", "signify", "suggestive"}
MODALS = {"may", "might", "could", "would", "must"}
NEG_WORDS = {"not", "n't", "never", "no", "neither", "nor", "none", "nobody", "nothing",
             "without"}
PERSON_NOUNS = {
    "man", "woman", "person", "people", "boy", "girl", "child", "children", "kid",
    "kids", "guy", "lady", "gentleman", "player", "skier", "surfer", "skateboarder",
    "rider", "worker", "men", "women", "baby", "toddler", "adult", "individual",
    "individuals", "mother", "father", "family", "friend", "friends", "couple",
    "student", "teacher", "chef", "cook", "officer", "driver", "pedestrian",
    "tourist", "customer", "athlete", "batter", "catcher", "pitcher", "umpire",
    "snowboarder", "biker", "cyclist", "motorcyclist", "spectator", "spectators",
    "crowd", "fan", "fans", "passenger", "passengers", "traveler", "teenager", "teen",
    "girls", "boys", "persons", "player", "players", "soldier", "police", "policeman",
    "vendor", "shopper", "audience", "group", "team", "member", "members", "bride",
    "groom", "husband", "wife", "son", "daughter", "brother", "sister", "infant",
    "youngster", "elderly", "senior", "staff", "employee", "doctor", "nurse",
    "farmer", "jockey", "referee", "goalie", "musician", "dancer", "singer", "artist",
}
MALE = {"man", "men", "boy", "boys", "guy", "gentleman", "father", "dad", "husband",
        "son", "brother", "king", "groom", "policeman", "sir"}
FEMALE = {"woman", "women", "girl", "girls", "lady", "mother", "mom", "wife",
          "daughter", "sister", "queen", "bride"}
BODY_PARTS = {"hand", "hands", "mouth", "arm", "arms", "lap", "head", "shoulder",
              "shoulders", "back", "finger", "fingers", "leg", "legs", "foot", "feet",
              "face", "neck", "ear", "ears", "chest", "knee", "knees", "eye", "eyes",
              "lips", "teeth", "palm", "wrist", "body", "side"}
SPATIAL_PREPS = {
    "on", "in", "at", "under", "underneath", "beneath", "below", "above", "over",
    "beside", "near", "behind", "between", "among", "amongst", "around", "across",
    "along", "alongside", "against", "through", "inside", "outside", "within",
    "toward", "towards", "into", "onto", "past", "by", "atop", "upon", "throughout",
    "off", "from", "down", "up", "beyond", "opposite", "amid", "amidst", "via",
    "before", "after", "facing", "to", "toward", "towards",
}
SYMMETRIC_SPATIAL = {"beside", "next to", "near", "close to", "adjacent to", "by",
                     "alongside", "between", "among", "amongst", "around", "with",
                     "opposite", "facing"}

MULTIWORD_HEADS = {"next": "to", "close": "to", "adjacent": "to", "near": "to",
                   "far": "from", "away": "from", "out": "of", "ahead": "of",
                   "because": "of", "instead": "of", "prior": "to"}
RELATION_VERB_PREPS = {
    ("look", "at"), ("stare", "at"), ("point", "at"), ("play", "with"),
    ("talk", "to"), ("talk", "with"), ("interact", "with"), ("reach", "for"),
    ("listen", "to"), ("smile", "at"), ("wave", "at"), ("chat", "with"),
    ("gaze", "at"), ("glance", "at"), ("speak", "to"), ("speak", "with"),
    ("feed", "on"), ("hold", "onto"), ("hold", "on"), ("cling", "to"),
    ("bite", "into"), ("aim", "at"), ("throw", "at"), ("bark", "at"),
    ("shake", "hands"), ("pose", "with"), ("work", "on"), ("lean", "on"),
}
SUBJECTIVE_VERBS = {"enjoy", "appreciate", "admire", "relax", "love", "like",
                    "celebrate", "savor", "cherish", "bond", "unwind", "embrace"}
SUBJECTIVE_OBJECTS = {"moment", "time", "experience", "day", "fun", "company",
                      "view", "outdoors", "environment", "atmosphere", "weather",
                      "meal", "game", "activity", "sunshine", "scenery", "life"}
OCR_VERBS = {"read", "say", "write", "label", "print", "display", "show", "spell",
             "feature", "bear", "inscribe", "mark", "state", "advertise"}


NON_ATTRIBUTE_ADJ = {"other", "another", "same", "different", "additional", "similar",
                     "next", "last", "first", "second", "third", "main", "various",
                     "several", "few", "many", "own", "such", "certain", "entire", "whole",
                     "only", "single", "individual", "respective", "visible", "present",
                     "overall", "numerous", "multiple", "possible", "likely", "apparent",
                     "actual", "real", "specific", "particular", "total", "more", "most",
                     "much", "little-known", "else", "remaining", "extra", "nearby"}
DISTINCT_MARKERS = {"another", "other", "second", "third", "different", "additional", "else"}
POSITION_ADJ = {"left", "right", "front", "back", "rear", "top", "bottom", "upper", "lower",
                "middle", "central", "far", "distant", "outer", "inner", "near", "closest",
                "nearest", "farthest", "foreground", "background", "left-hand", "right-hand"}
PART_NOUNS = {"part", "piece", "component", "section", "portion"}
LINKING_VERBS = {"feel", "sound", "smell", "taste", "remain", "become", "stay", "turn", "grow"}

ABSTRACT |= {"ability", "abilities", "skill", "skills", "talent", "technique", "expertise",
             "flow", "space", "touch", "nap", "rest", "break", "pitch", "swing", "ride",
             "trip", "journey", "match", "competition", "performance", "word", "words",
             "text", "lettering", "writing", "opportunity", "chance", "skill", "effort",
             "attempt", "task", "job", "work", "hobby", "sport", "routine", "outing",
             "gathering", "conversation", "discussion", "meeting", "celebration", "party"}

PARTITIVE = {"some", "others", "other", "one", "ones", "both", "many", "all", "several",
             "each", "most", "few", "none", "either", "neither", "any", "another", "rest"}

INTENT_VERBS = {"prepare", "try", "attempt", "want", "plan", "wait", "intend", "aim",
                "get", "begin", "start", "seek", "hope", "ready", "about", "going", "need"}

EXPLANATORY_VERBS = {"provide", "create", "add", "give", "contribute", "offer", "serve",
                     "lend", "evoke", "convey", "reflect", "emphasize", "enhance", "showcase",
                     "demonstrate", "exude", "bring", "set"}
HEDGE_INFER |= {"help", "allow", "enable", "ensure", "let", "make", "keep"}
VISIBILITY_ADJ = {"visible", "present", "seen", "located", "positioned", "placed",
                  "situated", "available", "displayed", "shown", "noticeable", "evident"}
RELATION_VERB_PREPS |= {("swing", "at"), ("consist", "of"), ("fill", "with"),
                        ("top", "with"), ("decorate", "with"), ("load", "with"),
                        ("cover", "with")}

LEXICALIZED = {"remote control", "french fries", "hot dog", "hot dogs", "stuffed animal",
               "living room", "dining room", "dining table", "cell phone", "mobile phone",
               "traffic light", "fire hydrant", "teddy bear", "ice cream", "polar bear",
               "grizzly bear", "golden retriever", "high chair", "wine glass", "coffee table",
               "electric guitar", "personal computer", "public transportation", "black bear",
               "brown bear", "white board", "flat screen", "tennis racket", "red light",
               "green light", "pickup truck", "single-decker bus", "double-decker bus"}
MASS_NOUNS = {"water", "grass", "food", "traffic", "snow", "sand", "dirt", "hair", "rice",
              "bread", "meat", "cheese", "sauce", "milk", "coffee", "tea", "juice", "wine",
              "beer", "smoke", "fog", "foliage", "furniture", "luggage", "equipment",
              "clothing", "greenery", "vegetation", "lettuce", "broccoli", "mud", "ice",
              "paper", "wood", "glass", "fruit", "salad", "pasta", "soup", "cake", "pizza",
              "cereal", "sky", "sunlight", "light", "ocean", "sea"}
SUBJECTIVE |= {"comfortable", "exciting", "challenging", "skillful", "skilled", "impressive", "fun-filled", "memorable"}

CLOTHING = {"shirt", "t-shirt", "tee", "dress", "uniform", "jacket", "coat", "suit", "sweater",
            "hoodie", "sweatshirt", "jersey", "blouse", "vest", "skirt", "pants", "trousers",
            "jeans", "shorts", "costume", "outfit", "gown", "robe", "apron", "tuxedo", "bikini",
            "swimsuit", "wetsuit", "overalls", "hat", "cap", "helmet", "scarf", "gloves", "boots",
            "shoes", "sneakers", "glasses", "sunglasses", "goggles", "clothes", "clothing", "attire",
            "gear", "raincoat", "parka", "kimono", "sari", "saree", "leggings", "pajamas", "tank-top"}

VISIBILITY_VERBS = {"see", "show", "display", "notice", "observe", "spot", "find"}
