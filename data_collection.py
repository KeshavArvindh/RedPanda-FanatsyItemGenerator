# fantasy_weapon_generator/data_collection.py
import pandas as pd
import requests
import random
from tqdm import tqdm
import json
import os

os.makedirs("data", exist_ok=True)

def collect_dnd_weapons():
    print("Collecting D&D weapons data...")
    base_url = "https://www.dnd5eapi.co/api/equipment-categories/weapon"
    
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Error fetching D&D API: {response.status_code}")
        return []
    
    data = response.json()
    weapons = []
    
    for item in tqdm(data.get("equipment", [])):
        weapon_url = f"https://www.dnd5eapi.co{item['url']}"
        weapon_response = requests.get(weapon_url)
        
        if weapon_response.status_code == 200:
            weapon_data = weapon_response.json()
            
            weapon = {
                "name": weapon_data.get("name", "Unknown Weapon"),
                "type": weapon_data.get("weapon_category", ""),
                "damage_type": weapon_data.get("damage", {}).get("damage_type", {}).get("name", ""),
                "damage_dice": weapon_data.get("damage", {}).get("damage_dice", ""),
                "description": weapon_data.get("desc", [""])[0] if weapon_data.get("desc") else "",
                "properties": [p.get("name") for p in weapon_data.get("properties", [])]
            }
            
            weapons.append(weapon)
    
    print(f"Collected {len(weapons)} D&D weapons")
    return weapons

def generate_fantasy_prefixes():
    """Generate fantasy-style prefixes for weapon names"""
    return [
        "Ancient", "Blazing", "Celestial", "Dread", "Eldritch", "Frost", "Gloom", 
        "Havoc", "Infernal", "Jade", "Kraken", "Lunar", "Mystic", "Nether", 
        "Onyx", "Phantom", "Quicksilver", "Rune", "Serpent", "Thunder", 
        "Umbral", "Venom", "Whisper", "Xenith", "Yttrium", "Zephyr",
        "Arcane", "Berserker's", "Chaos", "Dragon", "Ember", "Feral",
        "Ghost", "Hallowed", "Ivory", "Jinx", "Knight's", "Leviathan", 
        "Mithril", "Nova", "Oracle", "Primal", "Quartz", "Radiant",
        "Shadow", "Tempest", "Undying", "Vortex", "Warden's", "Aegis"
    ]

def generate_fantasy_suffixes():
    """Generate fantasy-style suffixes for weapon names"""
    return [
        "of Doom", "of the Ancients", "of Power", "of Twilight", "of the Void", 
        "of Souls", "of Dragons", "of Eternity", "of Frost", "of Flames",
        "of Destruction", "of Oblivion", "of the Eclipse", "of Reckoning", 
        "of the Fallen", "of Judgment", "of Shadows", "of Storms", 
        "of the Abyss", "of Vengeance", "of Despair", "of Glory",
        "of the Hunt", "of Darkness", "of Light", "of the Elements",
        "of Nightmares", "of Dreams", "of Conquest", "of Victory",
        "of Agony", "of Time", "of Salvation", "of the Cosmos"
    ]

def generate_base_weapons():
    """Generate base weapon types"""
    return [
        "Sword", "Axe", "Bow", "Staff", "Dagger", "Hammer", "Mace", 
        "Spear", "Wand", "Scythe", "Blade", "Claymore", "Falchion",
        "Glaive", "Halberd", "Katana", "Lance", "Longbow", "Rapier",
        "Saber", "Trident", "Whip", "Flail", "Crossbow", "Pike",
        "Maul", "Greataxe", "Greatsword", "Longsword", "Shortbow"
    ]

def generate_damage_types():
    """Generate damage types for weapons"""
    return [
        "Fire", "Ice", "Lightning", "Shadow", "Holy", "Arcane", "Poison",
        "Physical", "Chaos", "Necrotic", "Radiant", "Thunder", "Acid",
        "Psychic", "Force", "Void", "Blood", "Nature", "Wind"
    ]

def generate_weapon_properties():
    """Generate weapon properties"""
    return [
        "Enchanted", "Cursed", "Ancient", "Ethereal", "Soulbound", "Legendary",
        "Vampiric", "Vorpal", "Berserker", "Sharpened", "Balanced", "Runic",
        "Elemental", "Spectral", "Blessed", "Corrupted", "Sentient", "Mythical"
    ]

def generate_flavor_texts():
    """Generate flavor text templates for weapon descriptions"""
    return [
        "Forged in the depths of {place}, this {weapon_type} {characteristic}.",
        "Said to have belonged to {owner}, this {weapon_type} {power}.",
        "Legend says this {weapon_type} was {origin}. It {effect}.",
        "This {weapon_type} {characteristic} when {condition}.",
        "A {weapon_type} of {quality}, {effect} upon its enemies.",
        "Wielded long ago by {owner}, this {weapon_type} {power}.",
        "{owner}'s favored {weapon_type}, known to {effect}.",
        "This {weapon_type} {characteristic}, making it {quality}.",
        "A {weapon_type} {origin}, it {effect} with every strike.",
        "When {condition}, this {weapon_type} {power}.",
    ]

def generate_places():
    """Generate mythical places for weapon origins"""
    return [
        "the Eternal Forge", "Mount Doomspire", "the Abyssal Depths", 
        "Celestial Kingdoms", "the Shadow Realm", "Dragonheart Caverns",
        "the Astral Plane", "the Emerald Valley", "Kraken's Deep",
        "the Phoenix Peaks", "Elven Glades", "Dwarven Underhalls",
        "the Frozen Wastes", "Demon's Crucible", "the Starfall Isles"
    ]

def generate_owners():
    """Generate previous owners or creators of weapons"""
    return [
        "King Thoran", "Archmage Vexus", "the Dread Pirate Nex", 
        "Queen Elyndra", "Warlord Grommash", "the Demon Prince",
        "High Priestess Lunara", "the Last Dragon", "Emperor Valorian",
        "the Twilight Assassin", "Blacksmith Durin", "the Fey Queen",
        "Lich King Morthus", "Captain Silverhand", "Oracle Zephyrus"
    ]

def generate_characteristics():
    """Generate characteristics of weapons"""
    return [
        "pulses with arcane energy", "burns with eternal flames",
        "whispers dark secrets", "glows with divine light",
        "hungers for the blood of enemies", "freezes the air around it",
        "shifts its form slightly", "drains life from those it cuts",
        "radiates with celestial power", "leaves shadowy trails when swung",
        "hums an ancient melody", "sends shivers down one's spine",
        "feels impossibly light", "seems to predict its wielder's thoughts",
        "never requires sharpening", "vibrates in the presence of evil"
    ]

def generate_powers():
    """Generate special powers or abilities of weapons"""
    return [
        "can cut through any armor", "grants visions of the future",
        "strikes with the force of lightning", "never misses its target",
        "returns to the wielder's hand when thrown", "ignites enemies on contact",
        "freezes foes with a touch", "reveals hidden enemies",
        "grows stronger with each kill", "protects against dark magic",
        "amplifies magical abilities", "instills fear in enemies",
        "can cleave through stone", "slows time during combat",
        "binds the souls of those it slays", "emits blinding light on command"
    ]

def generate_origins():
    """Generate origin stories for weapons"""
    return [
        "crafted from a fallen star", "forged in dragon fire",
        "made from the bones of a god", "created by ancient elven smiths",
        "born from the tears of a goddess", "found in an ancient tomb",
        "pulled from the heart of a volcano", "gifted by a dying wizard",
        "recovered from the depths of the sea", "discovered in a hidden shrine",
        "stolen from the armory of demons", "constructed by clockwork gnomes",
        "grown from a magical seed", "carved from a single crystal",
        "assembled during a celestial alignment", "blessed by seven sages"
    ]

def generate_effects():
    """Generate effects that weapons produce"""
    return [
        "cleaves through reality itself", "leaves wounds that never heal",
        "causes foes to flee in terror", "drains the magic from spellcasters",
        "summons spectral allies in battle", "disrupts magical barriers",
        "creates illusions to confuse foes", "leaves trails of elemental energy",
        "heals the wielder with each strike", "reveals invisible creatures",
        "absorbs spells cast at the wielder", "marks enemies for hunting",
        "grows in power during the night", "binds victims in ghostly chains",
        "ignites in brilliant flames", "howls with the voices of past victims"
    ]

def generate_conditions():
    """Generate conditions that trigger weapon effects"""
    return [
        "bathed in moonlight", "covered in blood", "wielded with courage",
        "facing overwhelming odds", "in the hands of royalty", 
        "confronting ancient evils", "during the heat of battle",
        "under the light of the stars", "in complete darkness",
        "during a storm", "held by a true hero", "wielded with hatred",
        "facing demons or undead", "in the presence of magic",
        "wielded by one of pure heart", "during the wielder's greatest need"
    ]

def generate_qualities():
    """Generate qualities of weapons"""
    return [
        "unparalleled balance", "exceptional sharpness", "perfect weight",
        "ancient wisdom", "boundless rage", "deadly precision",
        "incredible durability", "haunting beauty", "mysterious origins",
        "unspeakable power", "divine blessing", "demonic corruption",
        "unfathomable lightness", "terrifying presence", "royal heritage"
    ]

def generate_synthetic_weapon_data(count=400):
    print(f"Generating {count} synthetic fantasy weapons...")
    
    prefixes = generate_fantasy_prefixes()
    suffixes = generate_fantasy_suffixes()
    base_weapons = generate_base_weapons()
    damage_types = generate_damage_types()
    properties = generate_weapon_properties()
    
    flavor_texts = generate_flavor_texts()
    places = generate_places()
    owners = generate_owners()
    characteristics = generate_characteristics()
    powers = generate_powers()
    origins = generate_origins()
    effects = generate_effects()
    conditions = generate_conditions()
    qualities = generate_qualities()
    
    weapons = []
    
    for _ in tqdm(range(count)):
        name_style = random.choice([
            "{prefix} {base}",
            "{base} {suffix}",
            "{prefix} {base} {suffix}",
            "{base} of {quality}",
            "The {prefix} {base}"
        ])
        
        name = name_style.format(
            prefix=random.choice(prefixes),
            base=random.choice(base_weapons),
            suffix=random.choice(suffixes),
            quality=random.choice(qualities).lower()
        )
        
        weapon_type = random.choice(base_weapons)
        damage_type = random.choice(damage_types)
        num_properties = random.randint(0, 3)
        weapon_properties = random.sample(properties, num_properties)
        
        template = random.choice(flavor_texts)
        description = template.format(
            place=random.choice(places),
            weapon_type=weapon_type.lower(),
            characteristic=random.choice(characteristics),
            owner=random.choice(owners),
            power=random.choice(powers),
            origin=random.choice(origins),
            effect=random.choice(effects),
            condition=random.choice(conditions),
            quality=random.choice(qualities).lower()
        )
        
        damage_effect = ""
        if damage_type == "Fire":
            damage_effect = " The weapon radiates heat and occasionally bursts into flames."
        elif damage_type == "Ice":
            damage_effect = " A thin layer of frost covers the weapon at all times."
        elif damage_type == "Lightning":
            damage_effect = " Small sparks dance across the surface when wielded."
        elif damage_type == "Shadow":
            damage_effect = " The weapon seems to absorb light from its surroundings."
        
        if random.random() < 0.7:  
            description += damage_effect
            
        weapon = {
            "name": name,
            "type": weapon_type,
            "damage_type": damage_type,
            "damage_dice": f"{random.randint(1, 3)}d{random.choice([4, 6, 8, 10, 12])}",
            "description": description,
            "properties": weapon_properties
        }
        
        weapons.append(weapon)
    
    return weapons

def format_dataset_for_training(weapons):
    formatted_data = []
    
    for weapon in weapons:
        if not weapon["description"]:
            continue
            
        prompt1 = f"Generate a name for a {weapon['damage_type'].lower()} {weapon['type'].lower()} with the following properties: {', '.join(weapon['properties']).lower()}"
        completion1 = weapon["name"]
        
        prompt2 = f"Write a description for a weapon called '{weapon['name']}'"
        completion2 = weapon["description"]
        
        prompt3 = f"Generate a fantasy weapon of type: {weapon['type'].lower()}"
        completion3 = f"Name: {weapon['name']}\nDamage Type: {weapon['damage_type']}\nDamage: {weapon['damage_dice']}\nProperties: {', '.join(weapon['properties'])}\nDescription: {weapon['description']}"
        
        formatted_data.append({"prompt": prompt1, "completion": completion1})
        formatted_data.append({"prompt": prompt2, "completion": completion2})
        formatted_data.append({"prompt": prompt3, "completion": completion3})
    
    return formatted_data

def main():
    dnd_weapons = collect_dnd_weapons()
    
    synthetic_weapons = generate_synthetic_weapon_data(400)
    
    all_weapons = dnd_weapons + synthetic_weapons
    
    print(f"Total weapons collected: {len(all_weapons)}")
    
    with open("data/weapons_raw.json", "w") as f:
        json.dump(all_weapons, f, indent=2)
    
    training_data = format_dataset_for_training(all_weapons)
    
    print(f"Total training examples: {len(training_data)}")
    
    with open("data/weapons_training.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    df = pd.DataFrame(training_data)
    df.to_csv("data/weapons_training.csv", index=False)
    
    print("Data collection and formatting complete!")

if __name__ == "__main__":
    main()