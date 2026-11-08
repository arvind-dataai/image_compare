"""Builds jewel_prompts.json: the 15 jewellery scene prompts (shared core text + one theme each) and their variation pools."""
import json
from pathlib import Path

CORE_TOP = """You are the art director of a premium jewellery brand's advertising studio. You are shown ONE photo of a jewellery product. It can be a ring, a necklace, a full set, earrings, bangles or a bracelet, a pendant, a chain, a maang tikka, an anklet, a nose pin, a brooch or any other piece of jewellery. Write ONE scene for a lifestyle advertising photograph of this exact jewellery. An image model will later paint the picture from your scene together with the photo, so every sentence must be concrete and directly usable."""

CORE_RULES = """WHAT NEVER CHANGES (the identity): the jewellery in the final picture must be exactly the jewellery in the photo: its design, shape and proportions, metal colour and finish, every stone, pearl or bead with its colour and count, its fine patterns and engraving, its size, the number of pieces, and any brand name, logo or engraving on it or on its own box or pouch. Never change, add or remove anything on it. Do not describe its design in detail; refer to it as "the jewellery from the reference photo" or by its type (the necklace set, the ring, the earrings).

WHAT YOU MAY CHANGE (the presentation): how it is placed, arranged, displayed and seen. The photo shows only one way it happened to be placed (often flat, bunched, tangled or on a mannequin). Do not repeat that arrangement. Give the jewellery a new, believable, beautifully arranged presentation that suits this scene, chosen the way a professional jewellery photographer shows this particular type of piece in real life. First decide the type of the piece from the photo, then use its proper support:
- Necklaces and sets: on a neck form or bust, laid in a smooth natural curve on a velvet or stone riser or a tray, or hanging from a stand or from the lid of an open box; never in a tangle. A set shows its matching earrings as a symmetrical pair beside it, each piece complete and separate.
- Earrings: hanging from an earring stand or card, or resting as a true mirror-image pair on a dish, a velvet pad or a stone.
- Rings: on a ring cone, in a slotted ring cushion, standing in an open ring box, or resting on a small stone or dish; never standing unsupported.
- Bangles and bracelets: stacked on a bangle rod or stand, on a wrist form, on a velvet roll or in a tray.
- Pendants, chains, maang tikka, anklets, nose pins, brooches and the rest: on the stand, form, cushion or card a real jeweller would use for that piece.

THE SCENE MUST BE PHYSICALLY REAL: the jewellery and every prop rest on real surfaces or hang from real supports, and obey gravity, weight and the natural behaviour of their material. Nothing floats, levitates, leans without support or balances impossibly. Show the jewellery at its true real-world size compared with the objects around it.

THE JEWELLERY IS THE HERO: it is the first thing the eye lands on: the clearest, sharpest, best-lit and richest in colour. The background is softer, calmer and less busy than the jewellery, and has real tone contrast with it (a gold piece is never on a gold or yellow background, a silver or white-stone piece never on a pale silver-grey one). Props are none to three, each smaller and quieter than the jewellery, never touching or hiding it, and they belong to the place.

NO PEOPLE, NO TEXT: no person, hand or body part and no mannequin face; worn jewellery is shown on a display form (a bust, neck form, hand or wrist form). No text, letters, numbers, price tags or logos other than the brand marks that are already on the jewellery or on its own box or pouch.

BOX AND BRANDING: a branded box, pouch or tag is part of the product, so if the photo shows one it may appear in the scene, neatly, sharp and unblurred, with its branding exactly as in the photo; never invent a new brand or logo. Plain plastic wrap, price stickers and cards are not part of the product and do not appear.

THE CREATIVE BRIEF: below your instructions the app adds a "CREATIVE BRIEF FOR THIS RUN" with specific choices for the setting, colours, light, camera, presentation and props, and a list of scenes already written for this photo with this prompt. The brief makes every run different, so follow it. Adjust its colours only as far as needed so the jewellery still stands out; if one item of the brief really cannot suit this piece, make the nearest believable choice. Never repeat a scene from the list, not even with small changes.

HOW TO WRITE: reply with ONE scene only, in plain text and in your own way and words, with no preamble and no alternatives. Make sure it tells the image model: the place and the exact spot in it, what the jewellery rests on, hangs from or is displayed on and how it is arranged, the background and surfaces in real materials and colours, the props and where they stand, the direction and quality of the light, the camera angle and distance, and the mood."""

SHARED = {
    "palettes": [
        "deep emerald green with warm cream", "midnight navy with soft champagne ivory", "dusty rose with warm ivory",
        "terracotta and sand", "charcoal grey with soft white", "burgundy wine with muted taupe",
        "sage green with oatmeal linen", "ink-blue with pale stone grey", "warm walnut brown with cream",
        "blush pink with soft grey", "forest green with antique white", "plum with soft lilac-grey",
        "teal with warm sand", "espresso brown with ivory", "slate blue with warm greige", "olive with warm cream",
    ],
    "lights": [
        "soft daylight from a window on the left", "soft daylight from a window on the right", "warm golden-hour side light",
        "gentle overcast diffused light", "warm lamp light in a dim room with a soft glow", "cool clear morning light from behind and above",
        "a soft spotlight on the jewellery with the room falling into shade", "light filtered through sheer curtains",
        "dappled light and soft leaf shadows", "warm candlelight accents with soft ambient light",
    ],
    "cameras": [
        "eye level, close, the jewellery filling about half the frame", "a 45-degree high angle at medium distance",
        "a low angle close to the surface with a shallow depth of field", "a three-quarter side view in close macro",
        "a wider view with the jewellery on the lower third and calm space above", "a close view slightly from the left with space on the right",
        "a close view slightly from the right with space on the left", "a straight-on front view at medium distance",
    ],
    "presentations": [
        "arranged flat and neatly on the surface", "draped or curved naturally over a soft form or a fold of fabric",
        "standing or hanging on a stand or a display form", "partly inside or beside its own open box, pouch or tray",
        "resting against a prop or on a raised step or riser", "following a gentle curve along the edge of a surface or tray",
    ],
    "props": [
        "no props at all", "one small flower stem in a slim vase", "a small ceramic or glass dish nearby",
        "a folded piece of silk or linen beside it", "a small unlit or softly lit candle", "the edge of a mirror or a small framed mirror behind it",
        "a closed book or a ribbon", "a small brass or stone object", "a few petals or leaves scattered nearby",
    ],
}

THEMES = [
 ("Bridal getting-ready morning", "The morning of a wedding, in the room where the bride gets ready: the jewellery laid out with care beside the other things of the day.",
  ["an ivory silk-draped dressing table with an antique mirror", "a heritage haveli bridal suite with a carved wooden vanity", "a hotel suite marble console beside a garment bag",
   "a velvet-topped trunk at the foot of a bed", "a low carved wooden stool scattered with rose petals", "a mirror-framed vanity with warm bulbs",
   "a carved wardrobe-top with a silk jewellery tray", "a window-lit side table with a brass bowl of flowers"]),
 ("Festive puja day at home", "A festival day at home (Diwali, Navratri, Akshaya Tritiya), jewellery set out with the puja items, ready to be worn and blessed.",
  ["a puja room with a brass diya and marigold garland on a carved wooden stool", "a silk-draped low table beside a brass thali", "a home temple shelf with soft diya glow",
   "a festive sideboard with fairy lights softly blurred behind", "an ornate cushion on a low wooden seat beside a rangoli-patterned tray", "a polished brass tray with flowers on a console",
   "a carved teak table with a lit oil lamp", "a silk-covered side table beside a vase of marigolds"]),
 ("Everyday dressing table", "The daily routine of getting ready: the jewellery on the owner's dressing table where it is chosen and worn every day.",
  ["a marble-top vanity with perfume bottles softly blurred behind", "a walnut dressing table by a window", "a ceramic jewellery tray beside a mirror",
   "a vanity with a makeup-brush holder at the side", "a bedroom dresser with a small plant", "a lacquered vanity with a leather jewellery roll",
   "a pale wood dressing table with a small framed mirror", "a console mirror shelf with a trinket dish"]),
 ("Gift reveal", "A gift for an anniversary, a birthday or a proposal: the moment the box is opened and the jewellery is first seen.",
  ["a dinner table with candles and wine glasses softly blurred", "a bedside table with a ribbon and a blank card", "an open velvet box on a silk cloth",
   "a gift box open on a sunlit windowsill", "an anniversary breakfast tray on a bed with linen", "a birthday table with a bouquet",
   "a festive gifting hamper on a sideboard", "a marble table with a rose and a ribbon"]),
 ("Boutique display", "A real jewellery boutique: the jewellery displayed the way a jeweller shows it to a customer.",
  ["a glass showcase with soft spotlights and a velvet bust", "a marble plinth in a luxury boutique with soft background lights", "a curved display counter with a velvet tray",
   "a mirrored shelf with a neck form", "a walnut display niche with a soft downlight", "a brass-and-glass tiered stand",
   "a velvet display pad in a vitrine", "a boutique window display on a raised stone step"]),
 ("Travel and getaway", "The jewellery on a trip: packed in a travel case and unpacked in a hotel or villa.",
  ["a leather jewellery travel case open on a hotel bedside table", "a luggage rack top with a folded silk scarf", "a resort balcony table with a sea-blue blur behind",
   "a villa terrace table with a straw hat", "an airport lounge side table", "an open suitcase tray with a velvet insert",
   "a hotel desk with a passport wallet", "a bedside table in a boutique hotel with a travel diary"]),
 ("Evening party", "Getting ready for an evening party or a cocktail night: the jewellery that finishes the outfit.",
  ["a dark lacquer vanity with a soft lamp", "a console with a small clutch bag and a perfume bottle", "a velvet armchair arm beside a lamp",
   "a mirrored hallway console", "a gala dressing-room table with a glass of champagne", "a dim lounge side table with a candle",
   "a satin-covered dressing stool beside a clutch", "a glass-top table in a penthouse with city lights blurred"]),
 ("Office and daily wear", "Everyday work jewellery: light, simple and part of the morning routine before leaving for the day.",
  ["a work desk by a window with a notebook and a coffee cup", "a leather desk tray beside a closed laptop", "an entrance console beside a handbag",
   "a key bowl on a hallway console", "a sunlit bookshelf ledge", "a marble coaster on a side table with a plant",
   "a bedside table in the morning beside sunglasses", "a wooden desk with a small desk plant"]),
 ("Sangeet and reception glam", "A wedding celebration backstage: the jewellery with the festive outfit and flowers of the event.",
  ["a green-room mirror with bulbs and a vase of flowers", "a chair arm draped with an embroidered dupatta", "a table with a flower garland and a brass dish",
   "a mandap-side table with marigolds", "a reception lounge table with soft fairy lights", "a lehenga-fabric fold on a cushioned bench",
   "a decorated vanity inside a wedding tent", "a carved table with an arrangement of roses"]),
 ("Heirloom and tradition", "A treasured family piece passed down through generations, kept in its old box or trunk.",
  ["an antique carved wooden jewellery box, open", "an old brass-bound trunk with a velvet lining", "an heirloom dresser with an embroidered velvet cloth",
   "an old book and spectacles on a teak table", "a grandmother's almirah shelf with a silk potli", "a teak table with a brass mirror",
   "a heritage verandah table in soft shade", "a sandalwood box on a carved stand"]),
 ("Morning light in a bedroom", "A calm morning at home or in a hotel suite: soft sun and a quiet, luxurious mood.",
  ["a bedroom windowsill with sheer curtains", "a bedside table with a folded linen throw", "a hotel suite breakfast tray",
   "a sunlit marble bathroom vanity", "a bedside table with a tea cup", "a vanity stool cushion by a bright window",
   "a wooden bench at the foot of a bed", "a dressing table in a bright airy room"]),
 ("Garden and terrace", "An outdoor daytime occasion: a garden brunch, a terrace lunch or a stroll through flowers.",
  ["a wrought-iron garden table with a flower vase", "a terrace cafe table with a lemonade glass softly blurred", "a stone balustrade with potted plants",
   "a wooden table with a linen runner in a garden", "a poolside table with a straw tray", "an orchard bench",
   "a rooftop planter ledge", "a garden table under a pergola"]),
 ("Editorial flat lay", "A styled fashion-magazine flat lay: the jewellery composed with fabric and flowers on one surface.",
  ["a silk scarf on a stone surface", "a folded saree or dupatta border with fresh flowers", "dried pampas grass on a linen surface",
   "velvet and linen fabrics layered together", "a ceramic dish and a single bloom on a marble slab", "a wooden tray with petals",
   "a leather-bound notebook and flowers", "a patterned silk in a calm tone beside a vase shadow"],
  ["directly from above, the jewellery centred", "directly from above, the jewellery slightly off-centre with calm space on one side",
   "from a high angle of about 70 degrees", "from above with a slight tilt and a soft focus fall-off at the edges"]),
 ("Seasonal celebration", "A festive occasion of the year, with the props of that festival around the jewellery.",
  ["a Diwali table with a diya and marigold petals", "an Eid table with dates and a brass lantern", "a Christmas table with fir branches and warm lights",
   "a Valentine's table with a rose and a candle", "a Karwa Chauth tray with a decorated thali and mehndi cones", "a Raksha Bandhan tray with a thali and a rakhi",
   "a New Year table in champagne tones with softly blurred lights", "a Holi-season table with soft pastel flowers"]),
 ("Gallery still life", "An art-directed luxury still life: the jewellery presented like an object in a gallery.",
  ["a stepped travertine plinth", "a black marble slab against a mirror", "a curved plaster niche",
   "a brushed-brass pedestal", "a fluted glass block beside a stone", "a sculpted sand-coloured plaster form",
   "a velvet-draped stone block", "a shallow stone tray on a dry stone ledge"]),
]


def build():
    out = []
    for i, t in enumerate(THEMES, start=1):
        title, usecase, settings = t[0], t[1], t[2]
        cams = t[3] if len(t) > 3 else None
        text = (CORE_TOP + "\n\nTHEME OF THIS PROMPT: " + title + ". " + usecase +
                " The scene must show this real use case or display method, and its place must be one in which this kind of jewellery is really used or shown.\n\n" + CORE_RULES)
        out.append({"id": i, "title": title, "summary": usecase, "prompt": text, "settings": settings, "cameras": cams})
    return {"shared": SHARED, "prompts": out}


if __name__ == "__main__":
    Path(__file__).resolve().parent.parent.joinpath("jewel_prompts.json").write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("written")
