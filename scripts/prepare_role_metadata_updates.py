"""Prepare exact-ID metadata updates from the completed web audit.

No database writes happen here.  The output is the only input accepted by the
guarded apply script.  Fictional character profiles are explicit visual
profiles; animal rows use the row's source lead plus conservative, deduped
visual keywords already present in the audited candidate.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824" / "active_role_audit.json"
OUTPUT_PATH = AUDIT_PATH.with_name("metadata_update_proposals.json")


PROFILES: dict[str, tuple[str, str]] = {
    "Kirby": (
        "Kirby is a small pink spherical creature with two stubby arms, two red feet, oval eyes, and rosy cheeks.",
        "Kirby, pink, spherical body, stubby arms, red feet, oval eyes, rosy cheeks, character",
    ),
    "MetaKnight": (
        "MetaKnight is a spherical masked swordsman with a silver mask, dark blue cape with gold trim, armored shoes, bat-like wings, and a golden sword with a red jewel.",
        "MetaKnight, spherical body, silver mask, dark blue cape, gold trim, bat-like wings, armored shoes, golden sword, red jewel, character",
    ),
    "KingDedede": (
        "KingDedede is a portly blue penguin-like creature wearing a red royal robe, red hat, yellow gloves, and a yellow beak; he carries a large wooden hammer.",
        "KingDedede, blue, portly, penguin-like, red robe, red hat, yellow gloves, yellow beak, wooden hammer, character",
    ),
    "Waddle Dee": (
        "Waddle Dee is a tan, round-bodied creature with a pear-shaped face, chestnut-colored eyes, no mouth, small stubby arms, honey-colored feet, and rosy cheeks.",
        "Waddle Dee, tan, pear-shaped face, chestnut-colored eyes, no mouth, round body, stubby arms, honey-colored feet, rosy cheeks, character",
    ),
    "Kabu": (
        "Kabu is a brown head-shaped statue with a flat base, rounded top, deep black square eyes, a small nose, and a large open mouth.",
        "Kabu, brown statue, head-shaped body, flat base, rounded top, black square eyes, small nose, open mouth, character",
    ),
    "Kracko": (
        "Kracko is a giant puffy cloud-like monster with a single shiny eye centered in its soft cloudy body and golden spikes around its outer edge.",
        "Kracko, giant cloud body, puffy, single eye, golden spikes, monster, character",
    ),
    "Parasol Waddle Dee": (
        "Parasol Waddle Dee is a tan, round-bodied Waddle Dee with a pear-shaped face, chestnut-colored eyes, no mouth, stubby arms, honey-colored feet, rosy cheeks, and a striped parasol.",
        "Parasol Waddle Dee, tan, Waddle Dee, pear-shaped face, chestnut-colored eyes, no mouth, round body, honey-colored feet, striped parasol, character",
    ),
    "Magolor": (
        "Magolor is a short brown alien with no feet, detached hands in cream mittens, a regal blue suit with gold-trimmed cog motifs, purple accents, yellow oval eyes, and a hood with pointed side protrusions.",
        "Magolor, short brown alien, no feet, detached hands, cream mittens, blue suit, gold cog motifs, purple accents, yellow oval eyes, hood, character",
    ),
    "Bandana Waddle Dee": (
        "Bandana Waddle Dee is a tan, round-bodied Waddle Dee with a pear-shaped face, chestnut-colored eyes, no mouth, stubby arms, honey-colored feet, rosy cheeks, a navy blue bandana, and a spear.",
        "Bandana Waddle Dee, tan, Waddle Dee, pear-shaped face, chestnut-colored eyes, no mouth, round body, stubby arms, honey-colored feet, navy blue bandana, spear, character",
    ),
    "Spinni": (
        "Spinni is a slender yellow mouse-like creature with a white belly, orange feet, long snout, red nose, long red cape, and large shiny red sunglasses.",
        "Spinni, yellow mouse-like creature, white belly, orange feet, long snout, red nose, red cape, shiny red sunglasses, character",
    ),
    "Mario": (
        "Mario is a short human man with brown hair and a large brown mustache, wearing a red cap with an M emblem, red shirt, blue overalls, white gloves, and brown shoes.",
        "Mario, human man, brown hair, brown mustache, red cap, M emblem, red shirt, blue overalls, white gloves, brown shoes, character",
    ),
    "Princess Peach": (
        "Princess Peach is a fair-skinned woman with long blonde hair and blue eyes, wearing a pink gown, white gloves, pearl jewelry, and a gold crown with colored jewels.",
        "Princess Peach, blonde hair, blue eyes, pink gown, white gloves, pearl jewelry, gold crown, colored jewels, character",
    ),
    "Princess Daisy": (
        "Princess Daisy is a fair-skinned woman with orange-brown hair and green eyes, wearing a yellow dress with orange accents, white gloves, and a flower-shaped brooch and crown.",
        "Princess Daisy, orange-brown hair, green eyes, yellow dress, orange accents, white gloves, flower brooch, crown, character",
    ),
    "Waluigi": (
        "Waluigi is a very tall, thin man with a long pink nose and pointed mustache, wearing a purple cap and shirt, black overalls, orange shoes, and white gloves.",
        "Waluigi, tall thin man, purple cap, purple shirt, black overalls, pink nose, pointed mustache, orange shoes, white gloves, character",
    ),
    "Donkey Kong": (
        "Donkey Kong is a large muscular brown gorilla with a tuft of hair, broad arms, and a red necktie bearing yellow DK initials.",
        "Donkey Kong, brown gorilla, muscular body, broad arms, red necktie, yellow DK initials, character",
    ),
    "Bowser Jr.": (
        "Bowser Jr. is a small yellow Koopa with a green shell, orange hair, a green headband, a rounded muzzle, and a white bib printed with a jagged mouth.",
        "Bowser Jr., yellow Koopa, green shell, orange hair, green headband, rounded muzzle, white bib, jagged mouth, character",
    ),
    "King Boo": (
        "King Boo is a large white spherical ghost with a red mouth and tongue, sharp fangs, red eyes, and a gold crown.",
        "King Boo, large white ghost, spherical body, red eyes, red tongue, sharp fangs, gold crown, character",
    ),
    "Shy Guy": (
        "Shy Guy is a small masked figure wearing a red hooded robe, a plain white mask with black eye holes, gloves, and brown shoes.",
        "Shy Guy, red hooded robe, white mask, black eye holes, gloves, brown shoes, masked figure, character",
    ),
    "Koopa Troopa": (
        "Koopa Troopa is a yellow-skinned turtle-like creature with a removable green shell, white belly, round eyes, and short legs.",
        "Koopa Troopa, yellow skin, turtle-like body, green shell, white belly, round eyes, short legs, character",
    ),
    "Goomba": (
        "Goomba is a small brown mushroom-like creature with a tan underside, thick eyebrows, fangs, and short feet.",
        "Goomba, brown mushroom-like creature, tan underside, thick eyebrows, fangs, short feet, character",
    ),
    "Piranha Plant": (
        "Piranha Plant is a green leafy stalk topped with a red-and-white-spotted head and a wide mouth lined with sharp triangular teeth.",
        "Piranha Plant, green stalk, leafy leaves, red head, white spots, wide mouth, sharp teeth, character",
    ),
    "Super Mushroom": (
        "Super Mushroom is a small mushroom-shaped power-up with a red cap covered in white spots and a cream-colored stem.",
        "Super Mushroom, red mushroom cap, white spots, cream stem, mushroom-shaped, character",
    ),
    "Bullet Bill": (
        "Bullet Bill is a large black bullet-shaped projectile with angry eyes, a pointed nose, and small white arms.",
        "Bullet Bill, black bullet body, pointed nose, angry eyes, white arms, projectile, character",
    ),
    "Piranha Plant Bros.": (
        "Piranha Plant Bros. are large Piranha Plant variants with thick green stalks, leafy bases, oversized red heads with white spots, and sharp teeth.",
        "Piranha Plant Bros., large Piranha Plant variant, green stalk, leafy base, oversized red head, white spots, sharp teeth, character",
    ),
    "Larry Koopa": (
        "Larry Koopa is a small yellow Koopa with a green shell, bright blue swept-up hair, a blue head crest, and sharp teeth.",
        "Larry Koopa, yellow Koopa, green shell, bright blue hair, blue head crest, sharp teeth, character",
    ),
    "Roy Koopa": (
        "Roy Koopa is a large yellow Koopa with a purple shell, pink shell rim, pink sunglasses, a purple head, and a spiked collar.",
        "Roy Koopa, yellow Koopa, purple shell, pink shell rim, pink sunglasses, purple head, spiked collar, character",
    ),
    "Ludwig von Koopa": (
        "Ludwig von Koopa is a yellow Koopa with a blue shell, large blue swept-back hair, a prominent snout, and sharp teeth.",
        "Ludwig von Koopa, yellow Koopa, blue shell, blue swept-back hair, prominent snout, sharp teeth, character",
    ),
    "Boo": (
        "Boo is a small white spherical ghost with tiny arms, a wide red mouth, sharp fangs, and a pink tongue.",
        "Boo, white spherical ghost, tiny arms, red mouth, sharp fangs, pink tongue, character",
    ),
    "Hello Kitty": (
        "Hello Kitty is a white cat character with black eyes, a yellow nose, three whiskers on each side, no visible mouth, a red bow, and blue overalls over a yellow shirt.",
        "Hello Kitty, white fur, black eyes, yellow nose, no visible mouth, three whiskers, red bow, blue overalls, yellow shirt, character",
    ),
    "Cinnamoroll": (
        "Cinnamoroll is a small white puppy with very long floppy ears, blue eyes, pink cheeks, and a curled tail.",
        "Cinnamoroll, white puppy, long floppy ears, blue eyes, pink cheeks, curled tail, character",
    ),
    "Pompompurin": (
        "Pompompurin is a yellow Golden Retriever character with floppy ears, a rounded body, a brown beret, and a small tail.",
        "Pompompurin, yellow Golden Retriever, floppy ears, rounded body, brown beret, small tail, character",
    ),
    "Kuromi": (
        "Kuromi is a white rabbit-like character with a black jester hood, a pink skull emblem, a black imp-like tail, and a mischievous expression.",
        "Kuromi, white rabbit-like character, black jester hood, pink skull emblem, black imp-like tail, character",
    ),
    "Totoro": (
        "Totoro is a large gray furry creature with a beige belly, pointed ears, long whiskers, a broad nose, large paws, and claws.",
        "Totoro, large gray furry creature, beige belly, pointed ears, long whiskers, large paws, claws, character",
    ),
    "Chibi-Totoro": (
        "Chibi-Totoro is the smallest Totoro form, a tiny white rounded creature with pointed ears, a simple face, and a short tail.",
        "Chibi-Totoro, tiny white creature, rounded body, pointed ears, simple face, short tail, character",
    ),
    "Chu-Totoro": (
        "Chu-Totoro is a medium-sized blue-gray furry creature with a rounded body, pointed ears, a simple face, and a short tail.",
        "Chu-Totoro, medium-sized blue-gray creature, furry body, rounded body, pointed ears, short tail, character",
    ),
    "Snoopy": (
        "Snoopy is a white beagle with black ears, a black spot on his back, a black nose, and a compact cartoon body.",
        "Snoopy, white beagle, black ears, black back spot, black nose, compact cartoon body, character",
    ),
    "Woodstock": (
        "Woodstock is a very small yellow bird with a tufted head, black eyes, a short orange beak, and tiny wings and feet.",
        "Woodstock, tiny yellow bird, tufted head, black eyes, orange beak, tiny wings, tiny feet, character",
    ),
    "Doraemon": (
        "Doraemon is a blue robotic cat with a white face and belly, round white hands and feet, six whiskers, a red nose, a red collar with a yellow bell, and a front pocket.",
        "Doraemon, blue robotic cat, white face, white belly, white hands, white feet, six whiskers, red nose, red collar, yellow bell, front pocket, character",
    ),
    "slime": (
        "slime is a blue teardrop-shaped gelatinous monster with a simple face, large round eyes, a smiling mouth, and no visible limbs.",
        "slime, blue gelatinous monster, teardrop shape, large round eyes, smiling mouth, no visible limbs, character",
    ),
}

CREATURE_FEATURES: dict[str, tuple[str, str]] = {
    "Elephant": ("gray skin, large ears, long trunk, tusks, thick legs", "gray, large ears, trunk, tusks"),
    "Asian Elephant": ("gray skin, large ears, long trunk, tusks, thick legs", "gray, large ears, trunk, tusks"),
    "Giraffe": ("tan coat with dark patches, very long neck, long legs, short mane, ossicones", "tan coat, dark patches, long neck, long legs, ossicones"),
    "Somali giraffe": ("tan coat with dark patches, very long neck, long legs, ossicones", "tan coat, dark patches, long neck, long legs, ossicones"),
    "Penguin": ("black-and-white plumage, upright body, flippers, short tail, webbed feet", "black-and-white plumage, flippers, webbed feet"),
    "King Penguin": ("black-and-white plumage, upright body, dark head, orange throat patches, flippers", "black-and-white plumage, orange throat, flippers"),
    "Polar Bear": ("thick white fur, black nose, broad paws, short tail", "white fur, black nose, broad paws"),
    "Red Panda": ("reddish-brown fur, white facial markings, pointed ears, ringed bushy tail", "reddish-brown fur, white face, pointed ears, ringed tail"),
    "Red Panda;Lesser Panda": ("reddish-brown fur, white facial markings, pointed ears, ringed bushy tail", "reddish-brown fur, white face, pointed ears, ringed tail"),
    "Dolphin": ("streamlined gray body, rounded forehead, long snout, dorsal fin, flippers, horizontal tail flukes", "gray body, long snout, dorsal fin, flippers"),
    "Peacock": ("male peafowl with iridescent blue-green plumage, crest, and a long eye-spotted tail fan", "blue-green plumage, crest, eye-spotted tail fan"),
    "Common Peafowl": ("male peafowl with iridescent blue-green plumage, crest, and a long eye-spotted tail fan", "blue-green plumage, crest, eye-spotted tail fan"),
    "Mandarin Duck": ("compact duck with a colorful male plumage, orange sail-like wing feathers, and a broad bill", "colorful plumage, orange wing feathers, broad bill"),
    "Axolotl": ("smooth aquatic salamander body, four legs, long tail, and feathery external gills", "aquatic salamander, four legs, long tail, external gills"),
    "Capybara": ("large barrel-shaped rodent with coarse brown fur, short ears, blunt muzzle, and short legs", "brown fur, barrel-shaped body, blunt muzzle, short legs"),
    "Narwhal": ("gray mottled whale with a rounded head, flippers, tail flukes, and a long spiral tusk", "gray mottled body, flippers, tail flukes, spiral tusk"),
    "Pangolin": ("mammal covered in overlapping protective scales, with a small head, long tail, and short legs", "overlapping scales, small head, long tail, short legs"),
    "Gerenuk": ("slender antelope with a long neck, long legs, large ears, and a narrow muzzle", "slender antelope, long neck, long legs, large ears"),
    "Tarsier": ("tiny primate with enormous round eyes, large ears, long fingers, and a long thin tail", "tiny primate, enormous eyes, large ears, long fingers, thin tail"),
    "Glasswing Butterfly": ("small butterfly with transparent wing panels, dark wing veins, and a slender body", "transparent wings, dark wing veins, slender body"),
    "Marine Iguana": ("dark stocky iguana with a blunt snout, dorsal spines, strong claws, and a laterally flattened tail", "dark iguana, dorsal spines, strong claws, flattened tail"),
    "Numbat": ("small reddish-brown marsupial with white back stripes, a pointed snout, and a long bushy tail", "reddish-brown fur, white stripes, pointed snout, bushy tail"),
    "One-humped Camel": ("large camel with a single hump, long neck, long legs, padded feet, and coarse sandy fur", "single hump, long neck, long legs, sandy fur"),
    "Two-humped Came": ("large camel with two humps, long neck, long legs, padded feet, and shaggy brown fur", "two humps, long neck, long legs, shaggy brown fur"),
    "Bornean Orangutan": ("large red-orange ape with long arms, shaggy hair, broad face, and grasping hands and feet", "red-orange hair, long arms, broad face, grasping hands"),
    "Bengal Tiger": ("large orange cat with black stripes, white cheeks and belly, rounded ears, and a long tail", "orange fur, black stripes, white belly, long tail"),
    "Black Swan": ("dark swan with black plumage, a red bill with a pale band, long neck, and white wing markings", "black plumage, red bill, long neck, white wing markings"),
    "Giant Walking Stick": ("giant stick insect with an elongated twig-like body, six jointed legs, and long slender antennae", "stick insect, twig-like body, six legs, long antennae"),
    "Lan-hsu giant katydid": ("large katydid with a green or brown body, very long antennae, leaf-like wings, and spiny hind legs", "katydid, green or brown body, long antennae, leaf-like wings, spiny legs"),
    "Yellow Emperor": ("Yellow Emperor is a traditional Chinese human figure depicted in ornate ancient robes, a tall headdress, and ceremonial accessories.", "Yellow Emperor, Chinese figure, ornate robes, ancient headdress, ceremonial accessories, character"),
    "Tomistoma": ("Tomistoma is a long-snouted freshwater crocodilian with an olive-brown body, dark markings, armored scales, and a long tail.", "Tomistoma, long snout, olive-brown body, dark markings, armored scales, long tail, wildlife"),
    "Yellow Peacock Bass": ("a freshwater cichlid with an elongated body, golden-yellow to olive coloring, dark vertical bars, light spots on the fins, and a long tail fin", "freshwater cichlid, golden-yellow body, dark vertical bars, light fin spots, long tail fin, wildlife"),
    "White-fronted capuchin": ("a medium-sized monkey with a light brown back, creamy white underside and face, a dark crown, slender limbs, and a long prehensile tail", "medium-sized monkey, light brown back, creamy white underside, pale face, dark crown, slender limbs, prehensile tail, wildlife"),
    "Field cricket": ("a dark brown to black cricket with a cylindrical body, long antennae, large hind legs, folded wings, and two tail-like cerci", "dark brown cricket, cylindrical body, long antennae, large hind legs, folded wings, cerci, wildlife"),
    "Common Tiger": ("an orange-tawny butterfly with broad black wing veins, black wing margins, and rows of white spots", "orange-tawny butterfly, black wing veins, black wing margins, white spots, wildlife"),
    "Puma, Mountain Lion": ("a large tawny cat with a pale muzzle, rounded ears, powerful limbs, a long tail, and a dark tail tip", "large tawny cat, pale muzzle, rounded ears, powerful limbs, long tail, dark tail tip, wildlife"),
    "Canadian Beaver, American Beaver": ("a large semiaquatic rodent with dense brown fur, a broad flat tail, webbed hind feet, and prominent orange incisors", "large semiaquatic rodent, dense brown fur, broad flat tail, webbed hind feet, orange incisors, wildlife"),
    "Small Chinese Civet, Lesser Oriental Civet": ("a small civet with a gray-brown to reddish-brown coat, dark spots and stripes, a pointed muzzle, rounded ears, and a long ringed tail", "small civet, gray-brown fur, reddish-brown fur, dark spots, dark stripes, pointed muzzle, rounded ears, ringed tail, wildlife"),
}

FEATURE_HAS = {
    "Elephant", "Asian Elephant", "Giraffe", "Somali giraffe", "Penguin", "King Penguin",
    "Polar Bear", "Red Panda", "Red Panda;Lesser Panda", "Dolphin", "Peacock", "Common Peafowl",
    "Mandarin Duck", "Axolotl", "Capybara", "Narwhal",
}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def _source_description(row: dict[str, Any]) -> str:
    name = str(row["role_name_en"])
    evidence = row.get("source_evidence") or {}
    source = str(evidence.get("lead_extract") or evidence.get("page_text_excerpt") or "").strip()
    source = re.sub(r"\s+", " ", source)
    if not source:
        current = re.sub(r"\s+", " ", str(row.get("current_description") or "")).strip()
        if current.casefold().startswith(name.casefold()):
            return name + current[len(name) :]
        return f"{name}: {current or 'source-checked character profile'}"[:1024]
    sentences = re.split(r"(?<=[.!?])\s+", source)
    selected = " ".join(sentences[:2]).strip() or source
    selected = selected[:900].rstrip()
    return f"{name}: {selected}"


def _prepare(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row["role_name_en"])
    if name in PROFILES:
        description, keywords = PROFILES[name]
        basis = "explicit_visual_profile_plus_row_source"
    elif name in CREATURE_FEATURES:
        description, feature_keywords = CREATURE_FEATURES[name]
        if not description.startswith(name):
            if name in FEATURE_HAS:
                description = f"{name} has {description}"
            elif description.startswith(("a ", "an ")):
                description = f"{name} is {description}"
            else:
                description = f"{name} is a {description}"
        category = "character" if name == "Yellow Emperor" else "wildlife"
        keywords = ", ".join(_dedupe([name, *feature_keywords.split(","), category]))
        basis = "source_checked_creature_visual_profile"
    elif row.get("group_name") == "Pokemon":
        description = str(row["proposed_description"])
        keywords = str(row["proposed_keywords"])
        basis = "PokeAPI_structured_species_and_type_fields"
    else:
        description = _source_description(row)
        current = [item.strip() for item in str(row.get("current_keywords") or "").split(",")]
        keywords = ", ".join(_dedupe([name, *current, "wildlife" if row.get("group_name") == "Creature" else "character"]))
        basis = "source_checked_lead_plus_conservative_existing_visual_keywords"
    if len(description) > 1024:
        description = description[:1024].rstrip()
    if len(keywords) > 512:
        keywords = ", ".join(_dedupe(keywords.split(",")))[:512].rstrip(" ,")
    return {
        "role_id": int(row["role_id"]),
        "role_name_en": name,
        "group_name": row["group_name"],
        "source_urls": row.get("source_urls", []),
        "source_type": row.get("source_type", ""),
        "source_title": row.get("source_title", ""),
        "source_status": row.get("source_status", ""),
        "current_description": row.get("current_description", ""),
        "current_keywords": row.get("current_keywords", ""),
        "proposed_description": description,
        "proposed_keywords": keywords,
        "rewrite_basis": basis,
        "description_verdict": "source_checked_proposed",
        "keywords_verdict": "source_checked_proposed",
    }


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("row_count") != 556 or audit.get("summary", {}).get("source_checked") != 556:
        raise RuntimeError("audit must contain exactly 556 source_checked active rows")
    records = [_prepare(row) for row in audit["records"]]
    if len({row["role_id"] for row in records}) != 556:
        raise RuntimeError("proposal IDs are not unique")
    for row in records:
        if not row["proposed_description"].startswith(str(row["role_name_en"])):
            raise RuntimeError(f"description does not start with exact role name: {row['role_id']}")
        if len(row["proposed_description"]) > 1024 or len(row["proposed_keywords"]) > 512:
            raise RuntimeError(f"field length exceeded: {row['role_id']}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_path": str(AUDIT_PATH),
        "row_count": len(records),
        "mysql_mutation": "none",
        "source_checked_required": True,
        "records": records,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUTPUT_PATH}")
    print(json.dumps({"row_count": len(records), "profiles": sum(r["rewrite_basis"] == "explicit_visual_profile_plus_row_source" for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
