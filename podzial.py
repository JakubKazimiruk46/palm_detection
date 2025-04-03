import os
import json
import random

# Konfiguracja
input_file = "annotations.json"
output_dir = "dataset"

# Wczytaj dane
print(f"Próba wczytania pliku: {input_file}")
try:
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Sprawdź czy plik zaczyna się od '{' - jeśli nie, dodaj nawiasy klamrowe
    if not content.strip().startswith('{'):
        content = "{" + content + "}"
    
    data = json.loads(content)
    print(f"Wczytano dane. Liczba próbek: {len(data)}")

    # Wypisz pierwsze kilka kluczy dla weryfikacji
    sample_keys = list(data.keys())[:3]
    print(f"Przykładowe klucze: {sample_keys}")
    print(f"Przykładowa próbka: {data[sample_keys[0]]}")
    
    # Podział danych
    all_ids = list(data.keys())
    random.shuffle(all_ids)
    
    # Oblicz liczby próbek
    total = len(all_ids)
    subsample_count = int(total * 0.1)
    test_count = int(total * 0.2)
    train_count = total - subsample_count - test_count
    
    # Podziel identyfikatory
    subsample_ids = all_ids[:subsample_count]
    test_ids = all_ids[subsample_count:subsample_count + test_count]
    train_ids = all_ids[subsample_count + test_count:]
    
    print(f"Podział danych: train={len(train_ids)}, test={len(test_ids)}, subsample={len(subsample_ids)}")
    
    # Zapisz podział do pliku (bez przetwarzania obrazów)
    split_info = {
        "train": train_ids,
        "test": test_ids,
        "subsample": subsample_ids
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split.json"), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Zapisano podział do pliku: {os.path.join(output_dir, 'split.json')}")
    
except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku {input_file}")
except json.JSONDecodeError as e:
    print(f"Błąd: Nieprawidłowy format JSON: {e}")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")