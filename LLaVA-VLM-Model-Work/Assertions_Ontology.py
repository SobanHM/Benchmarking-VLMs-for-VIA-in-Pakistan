ENTITY_ONTOLOGY = {
  # -------- Structures / Infrastructure --------
    "stairs":   {"type": "structure"},
    "step":     {"type": "structure"},
    "ramp":     {"type": "structure"},
    "door":     {"type": "structure"},
    "corridor": {"type": "structure"},
    "aisle":    {"type": "structure"},
    "wall":     {"type": "structure"},
    "rack":     {"type": "structure"},
    "shelf":    {"type": "structure"},
    "counter":  {"type": "structure"},

    # -------- Furniture --------
    "chair":   {"type": "furniture"},
    "table":   {"type": "furniture"},
    "sofa":    {"type": "furniture"},
    "bed":     {"type": "furniture"},
    "cabinet": {"type": "furniture"},

    # -------- Agents --------
    "person": {"type": "agent"},
    "people": {"type": "agent"},
    "child":  {"type": "agent"},

    # -------- Moveable Objects --------
    "cart": {"type": "movable_object"},
    "box":  {"type": "movable_object"},
    "crate":{"type": "movable_object"},
    "bowl": {"type": "static"},
    "bowls": {"type": "static"},

    # -------- Appliances / Overhead --------
    "fan":   {"type": "appliance"},
    "light": {"type": "appliance"},
    "tv":    {"type": "appliance"},

    # -------- Floor / Surface --------
    "rug":   {"type": "floor_item"},
    "spill": {"type": "movable_object"},

    # -------- Products (open‑vocab) --------
    "product": {"type": "product"}
}

PRODUCT_CATEGORIES = {
    "beverage": ["tea", "coffee", "green tea", "juice", "soft drink"],
    "snack": ["cookies", "biscuits", "chips"],
    "grocery": ["oil", "cereal", "custard", "jelly"],
    "toy": ["toy", "doll", "ball"],
    "promotion": ["discount", "off", "%", "sale"],
    "clothes" : ["jackets", "sweeters", "jeans", "shirts", "t-shirts","pants", "undergarments"]
}

SAFETY_DIMENSIONS = {
    "level_change": ["stairs", "step", "ramp", "drop"],
    "path_obstruction": ["chair", "table", "sofa", "bed", "cart", "box"],
    "path_geometry": ["narrow", "tight", "crowded", "blocked"],
    "dynamic_agents": ["person", "people", "child"],
    "surface_condition": ["slippery", "wet", "spill", "rug"],
    "visibility_issue": ["dark", "poor lighting", "glare"],
    "overhead_hazard": ["fan", "hanging light", "signboard"],
    "edge_risk": ["counter edge", "table corner"],
    "temporary_obstacle": ["crate", "cleaning bucket"],
    "navigation_conflict": ["queue", "checkout crowd", "crossing people"]
}

SAFETY_RULES = [
    {"entity": "stairs", "condition": "ahead",        "risk": "high"},
    {"entity": "chair",  "condition": "in_path",      "risk": "medium"},
    {"entity": "cart",   "condition": "moving",       "risk": "high"},
    {"entity": "person", "condition": "approaching",  "risk": "medium"},
    {"entity": "oil",    "condition": "on_floor",     "risk": "high"},
    {"entity": "rug",    "condition": "loose",        "risk": "medium"},
    {"entity": "fan",    "condition": "low_height",   "risk": "medium"},
    {"entity": "spill",  "condition": "on_floor",     "risk": "high"},
    {"entity": "bowl", "condition": "on_floor", "risk": "medium"},
    {"entity": "bowls", "condition": "on_floor", "risk": "medium"}
]


# critical for navigation and product locating
SPATIAL_TERMS = ["left", "right", "ahead", "behind", "front", "top", "bottom", "middle", "near", "far"]

# visual attributes useful for identifying products or safety
VISUAL_ATTRIBUTES = {
    "white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple",
    "large", "small", "tall", "short", "empty", "full", "open", "closed",
    "plastic", "glass", "metal", "wooden", "metallic"
}

NAVIGATION_VERBS = ["walk", "move", "turn", "proceed", "avoid", "stop"]

LEXICAL_SAFETY_TRIGGERS = ["stairs", "obstacle", "narrow", "crowded", "slippery", "blocked", "door"]

ALIASES = {"people": "person", "bottles": "bottle", "biscuits": "cookies", "rack": "shelf"}


ONTOLOGY_METADATA = {
  "version": "1.0",
  "release_date": "2026-01-16",
  "frozen_dimensions": True,
  "compatible_metrics": ["hallucination", "omission", "safety"]
}
