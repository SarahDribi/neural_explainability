# project_root/scripts/sum_names.py
import argparse
import os
from datetime import datetime

def count_letters_only(s: str) -> int:
    # Compte uniquement les caractères alphabétiques Unicode (accents inclus)
    return sum(ch.isalpha() for ch in s)

def main():
    parser = argparse.ArgumentParser(
        description="Somme du nombre de lettres de deux noms (par défaut: seulement lettres, accents inclus)."
    )
    parser.add_argument("name1", type=str, help="Premier nom")
    parser.add_argument("name2", type=str, help="Deuxième nom")
    parser.add_argument("--raw", action="store_true",
                        help="Utiliser la longueur brute (ne pas filtrer par lettres).")
    args = parser.parse_args()

    # Choix du mode de comptage
    if args.raw:
        len1 = len(args.name1)
        len2 = len(args.name2)
        mode = "raw"
    else:
        len1 = count_letters_only(args.name1)
        len2 = count_letters_only(args.name2)
        mode = "letters_only"

    total = len1 + len2

    # Localiser project_root à partir de l’emplacement du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Dossier experiments/script_name+timestamp
    script_name = os.path.splitext(os.path.basename(__file__))[0]  # "sum_names"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(project_root, "experiments", f"{script_name}+{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    # Écriture du fichier config.yaml
    # Écriture du log .txt
    log_path = os.path.join(exp_dir, "log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Script: {script_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Name 1: {args.name1} -> {len1}\n")
        f.write(f"Name 2: {args.name2} -> {len2}\n")
        f.write(f"Total: {total}\n")

    print(f"{total}")  # stdout simple
    print(f"(log enregistré dans {log_path})")

if __name__ == "__main__":
    main()
