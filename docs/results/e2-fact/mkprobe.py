import sys, json
T = ["The {}", "Yesterday we visited the {}", "I read a book about the {}", "Many people have seen the {}",
     "A photo of the {}", "My friend talked about the {}", "In the news: the {}", "He walked towards the {}"]
target = ["Eiffel Tower"]
donors = ["Colosseum", "Pantheon", "Trevi Fountain", "Roman Forum", "Spanish Steps"]
others = ["Statue of Liberty", "Golden Gate Bridge", "Big Ben", "Tower Bridge", "Sydney Opera House", "Taj Mahal", "Great Wall",
 "Kremlin", "Acropolis", "Parthenon", "Sagrada Familia", "Empire State Building", "White House", "Space Needle", "Hagia Sophia",
 "Blue Mosque", "Burj Khalifa", "Petronas Towers", "CN Tower", "Leaning Tower", "Alhambra", "Prado Museum", "British Museum",
 "Hermitage Museum", "Reichstag", "Charles Bridge", "Tokyo Tower", "Forbidden City", "Red Square", "Sphinx", "Great Pyramid",
 "Hoover Dam", "Lincoln Memorial", "Buckingham Palace", "Sistine Chapel", "Rialto Bridge", "Uffizi Gallery", "Arc de Triomphe",
 "Champs-Elysees", "Palace of Versailles", "Moulin Rouge", "Pompidou Centre", "Edinburgh Castle", "Stonehenge", "Machu Picchu",
 "Christ the Redeemer", "Table Mountain", "Angkor Wat", "Mount Fuji", "Grand Canyon", "Times Square", "Central Park",
 "Brooklyn Bridge", "Willis Tower", "Gateway Arch", "Alamo", "Pentagon", "Vatican", "Louvre Pyramid", "London Eye"]
rows = [(s, t.format(s), "target") for s in target for t in T] + [(s, t.format(s), "donor") for s in donors for t in T] \
     + [(s, t.format(s), "other") for i, s in enumerate(others) for t in T[i % 4::4]]
probe1 = open(sys.argv[1]).read().splitlines()[24:]          # the natural paragraphs written for E1
json.dump(rows, open(sys.argv[2] + "/rows.json", "w"))
open(sys.argv[2] + "/probe.txt", "w").write("\n".join([r[1] for r in rows] + probe1) + "\n")
print(len(rows), "subject prompts +", len(probe1), "paragraphs")
