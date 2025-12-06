#!/usr/bin/env python3
"""
Syntetyczny generator danych PII do LoRA/NER (offline).
Tworzy jednocześnie tekst z placeholderami oraz tekst z wypełnionymi danymi
+ listę spanów (start, end, label), aby łatwo budować złoty standard.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# --- Słowniki bazowe (rozszerz wedle potrzeb) ---
DICTS = {
    "names_f": [
        "Anna",
        "Maria",
        "Ewa",
        "Katarzyna",
        "Julia",
        "Alicja",
        "Magdalena",
        "Zofia",
        "Barbara",
        "Joanna",
        "Agnieszka",
        "Marta",
    ],
    "names_m": [
        "Jan",
        "Piotr",
        "Paweł",
        "Michał",
        "Adam",
        "Krzysztof",
        "Tomasz",
        "Andrzej",
        "Łukasz",
        "Mateusz",
        "Jakub",
        "Maciej",
    ],
    "surnames": [
        "Kowalski",
        "Nowak",
        "Wiśniewski",
        "Wójcik",
        "Krawczyk",
        "Lewandowski",
        "Zieliński",
        "Szymański",
        "Woźniak",
        "Dąbrowski",
        "Mazur",
        "Kubiak",
    ],
    "cities": [
        "Warszawa",
        "Kraków",
        "Gdańsk",
        "Wrocław",
        "Poznań",
        "Łódź",
        "Lublin",
        "Szczecin",
        "Bydgoszcz",
        "Białystok",
        "Katowice",
        "Rzeszów",
    ],
    "streets": [
        "Długa",
        "Krótka",
        "Słoneczna",
        "Lipowa",
        "Polna",
        "Mickiewicza",
        "Kościuszki",
        "Szkolna",
        "Ogrodowa",
        "Leśna",
        "Kwiatowa",
        "Sportowa",
    ],
    "health": ["nadciśnienie", "cukrzyca", "astma", "depresja", "migrena", "choroba serca", "alergia"],
    "religion": ["katolicyzm", "prawosławie", "protestantyzm", "islam", "judaizm", "ateizm"],
    "political": ["konserwatywne", "liberalne", "centrowe", "socjalne", "wolnorynkowe", "proekologiczne"],
    "relative": ["matka", "ojciec", "siostra", "brat", "syn", "córka", "żona", "mąż", "babcia", "dziadek"],
    "job_title": ["inżynier", "nauczyciel", "lekarz", "programista", "analityk", "kierownik", "sprzedawca", "pielęgniarka", "prawnik", "psycholog"],
    "company": ["ABC Sp. z o.o.", "TechPol", "Medica SA", "FinServ", "EduPlus", "LogiTrans"],
    "sex": ["kobieta", "mężczyzna", "niebinarna"],
    "ethnicity": ["polska", "ukraińska", "białoruska", "litewska", "rosyjska", "niemiecka"],
    "sexual_orientation": ["heteroseksualna", "homoseksualna", "biseksualna", "aseksualna"],
    "school_name": ["LO nr 1 w Warszawie", "Technikum Elektroniczne w Gdańsku", "SP 12 w Krakowie", "ZS nr 3 w Poznaniu", "Politechnika Warszawska", "Uniwersytet Gdański"],
    "username": ["janek", "ania_k", "user123", "devpl", "ml_ops", "ner_master"],
}

CITY_LOC_MAP = {
    "Warszawa": "Warszawie",
    "Kraków": "Krakowie",
    "Gdańsk": "Gdańsku",
    "Wrocław": "Wrocławiu",
    "Poznań": "Poznaniu",
    "Łódź": "Łodzi",
    "Lublin": "Lublinie",
}

# --- Generatory wartości ---

def gen_pesel() -> str:
    digits = [str(random.randint(0, 9)) for _ in range(10)]
    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    checksum = (10 - sum(int(d) * w for d, w in zip(digits, weights)) % 10) % 10
    return "".join(digits) + str(checksum)


def gen_phone() -> str:
    base = "".join(str(random.randint(0, 9)) for _ in range(9))
    return random.choice([
        "+48 " + base,
        "+48 " + f"{base[:3]} {base[3:6]} {base[6:]}",
        f"{base[:3]}-{base[3:6]}-{base[6:]}",
    ])


def gen_iban() -> str:
    body = "".join(str(random.randint(0, 9)) for _ in range(26))
    return "PL" + body


def gen_document() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return random.choice(letters) + random.choice(letters) + str(random.randint(100000, 999999))


def gen_credit_card() -> str:
    digits = "".join(str(random.randint(0, 9)) for _ in range(16))
    if random.random() < 0.5:
        return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}"
    return f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"


def gen_date() -> str:
    year = random.randint(1950, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    sep = random.choice(["-", ".", "/"])
    return f"{day:02d}{sep}{month:02d}{sep}{year}"


def gen_dob() -> str:
    year = random.randint(1930, 2015)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    sep = random.choice(["-", ".", "/"])
    return f"{day:02d}{sep}{month:02d}{sep}{year}"


def gen_username(name: str, surname: str) -> str:
    base = random.choice([name, surname]).lower()
    suffix = random.choice([str(random.randint(1, 999)), "", "_pl", "2024"])
    return base + suffix


def gen_secret() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"  # noqa: E501
    length = random.randint(10, 16)
    return "".join(random.choice(alphabet) for _ in range(length))


def gen_email(name: str, surname: str) -> str:
    return f"{name.lower()}.{surname.lower()}{random.randint(1,99)}@example.com"


def build_address(city: str, street: str) -> str:
    nr = random.randint(1, 120)
    apt = random.choice(["", f"/{random.randint(1,60)}"])
    postal = f"{random.randint(10,99)}-{random.randint(100,999)}"
    return f"ul. {street} {nr}{apt} {postal} {city}"


def city_loc(city: str) -> str:
    return CITY_LOC_MAP.get(city, city + "ie")


def surname_gen(surname: str) -> str:
    return surname + ("y" if surname.endswith("a") else "a")


# --- Szum / noise ---

def add_noise(text: str) -> str:
    # drop diacritics occasionally
    if random.random() < 0.3:
        text = text.translate(str.maketrans("łśążźćńóŁŚĄŻŹĆŃÓ", "lsazzcnoLSAZZCNO"))
    # random character swap
    if random.random() < 0.2 and len(text) > 3:
        i = random.randint(0, len(text) - 2)
        text = text[:i] + random.choice("qwertyuiopasdfghjklzxcvbnm") + text[i + 1 :]
    # leet / OCR-like confusions
    if random.random() < 0.2:
        text = text.replace("0", random.choice(["O", "0"])).replace("1", random.choice(["l", "I", "1"]))
        text = text.replace("5", random.choice(["S", "5"]))
    # stray punctuation
    if random.random() < 0.15 and len(text) > 5:
        pos = random.randint(1, len(text) - 2)
        text = text[:pos] + random.choice(["#", "*", "/", "_"]) + text[pos:]
    return text


# --- Render z placeholderów do tekstu + spany ---

def render_with_spans(template: str, values: Dict[str, str]) -> Tuple[str, List[Dict]]:
    out = []
    spans: List[Dict] = []
    label_map = {
        "name": "name",
        "surname": "surname",
        "surname_gen": "surname",
        "city": "city",
        "city_loc": "city",
        "address": "address",
        "phone": "phone",
        "email": "email",
        "pesel": "pesel",
        "bank_account": "bank-account",
        "document_number": "document-number",
        "health": "health",
        "religion": "religion",
        "political": "political-view",
        "relative": "relative",
        "job_title": "job-title",
        "company": "company",
        "age": "age",
        "sex": "sex",
        "ethnicity": "ethnicity",
        "sexual_orientation": "sexual-orientation",
        "school_name": "school-name",
        "credit_card": "credit-card-number",
        "username": "username",
        "secret": "secret",
        "date_of_birth": "date-of-birth",
        "date": "date",
    }
    cursor = 0
    for m in re.finditer(r"\{([a-z_\-]+)\}", template):
        start, end = m.span()
        # tekst przed placeholderem
        out.append(template[cursor:start])
        key = m.group(1)
        val = values[key]
        span_start = sum(len(x) for x in out)
        out.append(val)
        span_end = span_start + len(val)
        label = label_map.get(key)
        if label:
            spans.append({"start": span_start, "end": span_end, "label": label})
        cursor = end
    out.append(template[cursor:])
    text = "".join(out)
    return text, spans


# --- Templatey (placeholdery) ---
TEMPLATES = {
    "urzad": [
        "Nazywam się {name} {surname}, PESEL {pesel}, mieszkam pod adresem {address}, tel. {phone}, email {email}.",
        "Dane wnioskodawcy: {name} {surname}, {document_number}, zam. {address}, miasto {city}.",
        "Zatrudniony jako {job_title} w firmie przy {address}, kontakt {phone}, {email}.",
        "W załączniku oświadczenie: {name} {surname} ({religion}), zam. {address}, PESEL {pesel}, tel {phone}.",
        "Wniosek o odpis: {name} {surname}, dokument {document_number}, adres korespondencyjny {address}, e-mail {email}.",
        "Dane darczyńcy: {name} {surname}, konto {bank_account}, adres {address}, miasto {city}.",
        "Decyzja administracyjna: {name} {surname}, PESEL {pesel}, adres {address}, kontakt {phone}; pełnomocnik {relative}.",
        "Zażalenie: {name} {surname_gen} wnioskuje o ponowne rozpatrzenie sprawy, dokument {document_number}, zam. {address} w {city_loc}.",
        "Protokół: obecni {name} {surname} oraz {relative}; miejsce {address}, miasto {city}, kontakt {email}.",
        "Zaświadczenie: {name} {surname}, PESEL {pesel}, adres {address}, wniosek złożony przez {relative}.",
        "Rejestracja: {name} {surname}, dokument {document_number}, miasto {city_loc}, telefon {phone}, e-mail {email}.",
        "Mandat: {name} {surname}, {document_number}, adres {address}, kontakt {phone}; płatność z konta {bank_account}.",
        "Przekazanie sprawy: {name} {surname}, reprezentuje go {relative}, adres {address}, miasto {city}.",
        "Wezwanie do uzupełnienia: {name} {surname_gen}, PESEL {pesel}, adres {address}, termin 7 dni.",
            "Potwierdzenie odbioru: {name} {surname}, adres {address}, odbiorca {relative}, telefon {phone}.",
            "Zawiadomienie: {name} {surname}, sygnatura {document_number}, miasto {city_loc}, mail {email}.",
        "Deklaracja: {name} {surname}, płeć {sex}, etniczność {ethnicity}, adres {address}, data urodzenia {date_of_birth}.",
        "Formularz: {name} {surname}, orientacja {sexual_orientation}, dokument {document_number}, miasto {city}, kontakt {phone}.",
        "Wniosek meldunkowy: {name} {surname}, data {date}, adres {address}, PESEL {pesel}, płeć {sex}.",
    ],
    "med": [
        "Pacjent {name} {surname} (PESEL {pesel}) zgłasza {health}. Kontakt: {phone}, {email}.",
        "W wywiadzie: {health}, religia {religion}. Lekarz prowadzący {name} {surname_gen}.",
        "Skierowanie: {name} {surname}, adres {address}, tel {phone}, mail {email}, choroby współistniejące: {health}.",
        "Opis wizyty: {name} {surname}, {age} lat, PESEL {pesel}, przyjmuje leki na {health}, kontakt {phone}.",
        "Recepta: pacjent {name} {surname}, pesel {pesel}, adres {address}, telefon {phone}; rozpoznanie {health}.",
        "Historia choroby: {name} {surname}, {age} lat, {health}; lekarz {relative} zalecił kontrolę w {city_loc}.",
        "Teleporada: {name} {surname}, {age} lat, objawy {health}, tel {phone}, mail {email}.",
        "Skierowanie na badania: {name} {surname}, PESEL {pesel}, adres {address}, laboratorium w {city_loc}.",
        "Karta pacjenta: {name} {surname}, {age} lat, kontakt {phone}, osoba do powiadomienia {relative}.",
        "Rejestracja na szczepienie: {name} {surname}, pesel {pesel}, telefon {phone}, punkt w {city_loc}.",
            "Skarga pacjenta: {name} {surname}, {health}, adres {address}, kontakt {email}, {phone}.",
            "Konsultacja specjalistyczna: {name} {surname}, skierowanie nr {document_number}, szpital w {city}.",
        "Historia: {name} {surname}, data ur. {date_of_birth}, płeć {sex}, etniczność {ethnicity}, choroba {health}.",
    ],
    "czat": [
        "Hej, tu {name}, tel {phone}, pisz na {email}, mieszkam w {city}.",
        "Spotkajmy się na {address} w {city_loc}, dzwoń {phone}.",
        "Mój {relative} mieszka w {city}, a ja pracuję jako {job_title} w {city_loc}.",
        "Siema, jestem w {city}, pracuję w {company}, złap mnie na {phone} albo {email}.",
        "Wpadnij jutro: {address}, {city}. Jak coś, numer {phone}.",
        "A: gdzie jesteś? B: w {city_loc}, w biurze {company}, pisz na {email}.",
        "Spotkanie o 19:00 przy {address}, mam tel {phone}, weź {relative} jeśli może.",
        "Potrzebuję pomocy w {city}: dzwoń {phone}, pisz {email}, jestem u {relative}.",
        "Jadę do {city_loc}, nocuję na {address}; jak coś, telefon {phone}.",
        "Pracuję zdalnie dla {company} z {city}, kontakt {email}, {phone}.",
        "A: podasz dane? B: {name} {surname}, {city}, {phone}, {email}, {job_title} w {company}.",
            "Potrzebuję noclegu w {city_loc}; pisz {email}, dzwoń {phone}, mam adres {address}.",
            "A: kto to? B: {relative} {surname_gen}, mieszka w {city}, tel {phone}.",
        "Nick: {username}, miasto {city}, mail {email}, hasło {secret} (nie podawaj dalej!).",
    ],
    "wyciek": [
        "Wyciekły dane: {name} {surname}, PESEL {pesel}, adres {address}, tel {phone}, email {email}, konto {bank_account}.",
        "Incydent: {name} {surname} ({political}), dokument {document_number}, zam. {address}.",
        "RODO alert: {name} {surname}, telefon {phone}, mail {email}, adres {address}, poglądy {political}.",
        "Zgłoszenie do UODO: {name} {surname}, konto {bank_account}, dowód {document_number}, miasto {city}.",
        "Notatka incydentu: {name} {surname}, PESEL {pesel}, adres {address}, powiązany {relative}; zgłoszono do inspektora.",
        "Raport DLP: wykryto {email}, {phone}, {address} należące do {name} {surname} w pliku eksportu.",
        "Alert SIEM: rekord {document_number}, {bank_account}, właściciel {name} {surname}, lokalizacja {city_loc}.",
        "Inspektor: sprawa {name} {surname_gen}, dane {pesel}, {email}, {phone}, naruszenie dotyczy {relative}.",
            "Log zdarzenia: uzytkownik {email}, IP z {city}, dane {bank_account}, wlasciciel {name} {surname}.",
            "Analiza incydentu: {document_number}, {name} {surname}, tel {phone}, polityka {political}.",
        "Wyciek loginów: {username}, hasło {secret}, karta {credit_card}, adres {address}.",
        "Eksport bazy: {name} {surname}, PESEL {pesel}, karta {credit_card}, mail {email}, telefon {phone}.",
    ],
    "praca": [
        "Umowa o pracę: {name} {surname}, stanowisko {job_title}, adres {address}, kontakt {phone}, {email}.",
        "Referencje dla {name} {surname_gen}, pracował jako {job_title} w {city_loc}.",
        "Rekrutacja: {name} {surname}, aplikacja na {job_title} w {company}, telefon {phone}, mail {email}.",
        "Kandydat {name} {surname}, PESEL {pesel}, adres {address}, firma {company}, rola {job_title}.",
        "Lista płac: {name} {surname}, {job_title} w {company}, konto {bank_account}, miasto {city_loc}.",
        "Oświadczenie BHP: {name} {surname}, dokument {document_number}, tel {phone}, zatrudniony w {company}.",
        "Onboarding: {name} {surname}, {job_title}, laptop wysłać na {address}, tel {phone}, mail {email}.",
        "Delegacja: {name} {surname}, firma {company}, miasto docelowe {city}, kontakt {phone}.",
        "Umowa zlecenie: {name} {surname_gen}, {job_title}, rachunek na {bank_account}, miasto {city}.",
            "ZUS ZUA: {name} {surname}, PESEL {pesel}, adres {address}, firma {company}, tel {phone}.",
            "Wypowiedzenie: {name} {surname_gen}, stanowisko {job_title}, miasto {city_loc}, kontakt {email}.",
        "Dostęp VPN: użytkownik {username}, hasło {secret}, firma {company}, miasto {city}.",
    ],
    "szkola": [
        "Uczeń {name} {surname}, klasa 3b, mieszka w {city}, kontakt {phone}, email {email}.",
        "Rodzic ({relative}) {name} {surname} z {city_loc} prosi o spotkanie; tel {phone}.",
        "Stypendium: {name} {surname}, adres {address}, PESEL {pesel}, szkoła w {city_loc}.",
        "Zgoda rodzica: {name} {surname}, dokument {document_number}, dziecko {relative} w {city_loc}.",
        "Dziennik uwag: {name} {surname}, {age} lat, zgłoszono kontakt {relative} tel {phone}.",
        "Korespondencja szkolna do {name} {surname_gen}, adres {address}, e-mail {email}, miasto {city}.",
        "Lista uczniów: {name} {surname}, PESEL {pesel}, opiekun {relative}, miasto {city}.",
        "Rekrutacja do szkoły: {name} {surname}, {age} lat, adres {address}, kontakt {phone}, mail {email}.",
        "Wycieczka: {name} {surname}, klasa 2a, zgoda {relative}, telefon alarmowy {phone}.",
            "Plan lekcji: uczen {name} {surname}, kontakt {relative}, miasto {city}, mail {email}.",
            "Rekrutacja do konkursu: {name} {surname}, wiek {age}, opiekun {relative}, telefon {phone}.",
        "Karta szkoły: {name} {surname}, szkoła {school_name}, miasto {city}, email {email}, PESEL {pesel}.",
    ],
    "dialog": [
        "A: przedstaw się. B: {name} {surname}, pracuję w {company}, tel {phone}, mail {email}.",
        "A: gdzie mieszkasz? B: {address} w {city_loc}, PESEL {pesel}.",
        "A: kogo mam powiadomić? B: {relative}, numer {phone}, adres {address}.",
        "A: kim jesteś? B: {job_title} w {company}, mieszkam w {city}, dokument {document_number}.",
        "A: potrzebuję kontaktu. B: {email}, {phone}, {address}, na nazwisko {surname_gen}.",
            "A: masz dokumenty? B: {document_number}, {bank_account}, adres {address}, miasto {city_loc}.",
            "A: kogo reprezentujesz? B: {relative}, {name} {surname}, kontakt {email}.",
    ],
    "ogloszenie": [
        "Ogłoszenie: szukam {job_title} w {city}, kontakt {phone}, {email}, firma {company}.",
        "Zaginął dokument {document_number} należący do {name} {surname}, adres {address}, proszę o kontakt {phone}.",
        "Zbiórka: {name} {surname}, konto {bank_account}, miasto {city}, organizuje {relative}.",
        "Sprzedam: kontakt {phone}, {email}, właściciel {name} {surname}, odbiór {address}.",
        "Poszukiwany świadek: {name} {surname}, tel {phone}, adres {address}, zdarzenie w {city_loc}.",
        "Oficjalne info: {company} zatrudni {job_title} w {city}, CV na {email}, tel {phone}.",
            "Przetarg: {company} poszukuje {job_title}, termin zgłoszeń {document_number}, kontakt {email}.",
            "Komunikat: {name} {surname} zmienia adres na {address}, nowy telefon {phone}.",
        "Ogłoszenie szkolne: {school_name} szuka nauczyciela {job_title}, kontakt {email}, miasto {city}.",
    ],
    "formal": [
        "Pismo: {name} {surname}, PESEL {pesel}, adres {address}, sygnatura {document_number}, kontakt {email}.",
        "Załącznik do akt: {name} {surname_gen}, {political}, telefon {phone}, miasto {city_loc}.",
        "Wniosek formalny: {name} {surname}, {bank_account}, pełnomocnik {relative}, adres {address}.",
            "Aneks: {name} {surname}, konto {bank_account}, adres {address}, data {document_number}, telefon {phone}.",
            "Notatka służbowa: {name} {surname}, {job_title}, firma {company}, miasto {city_loc}.",
        "Zaświadczenie o studiach: {name} {surname}, szkoła {school_name}, miasto {city}, tel {phone}.",
        "Oświadczenie RODO: {name} {surname}, data ur. {date_of_birth}, karta {credit_card}, e-mail {email}.",
    ],
    "paragraf": [
        "Paragraf 1: {name} {surname} zobowiązuje się do płatności z konta {bank_account}; adres {address}.",
        "Paragraf 2: strony {name} {surname} oraz {relative} ustalają kontakt {phone}, {email}.",
        "Paragraf 3: dokument {document_number} obowiązuje w {city_loc}.",
            "Paragraf 4: {name} {surname_gen} odpowiada za {relative}, zam. {address}, tel {phone}.",
            "Paragraf 5: dane {pesel}, {email} przechowywane zgodnie z regulaminem; właściciel {name} {surname}.",
        "Paragraf 6: dane {credit_card} oraz {document_number} zaszyfrowane; użytkownik {username}.",
    ],
}


def sample_values(dicts: Dict[str, List[str]]) -> Dict[str, str]:
    is_female = bool(random.getrandbits(1))
    name = random.choice(dicts["names_f" if is_female else "names_m"])
    surname = random.choice(dicts["surnames"])
    city = random.choice(dicts["cities"])
    street = random.choice(dicts["streets"])
    return {
        "name": name,
        "surname": surname,
        "surname_gen": surname_gen(surname),
        "city": city,
        "city_loc": city_loc(city),
        "address": build_address(city, street),
        "phone": gen_phone(),
        "email": gen_email(name, surname),
        "pesel": gen_pesel(),
        "bank_account": gen_iban(),
        "document_number": gen_document(),
        "health": random.choice(dicts["health"]),
        "religion": random.choice(dicts["religion"]),
        "political": random.choice(dicts["political"]),
        "relative": random.choice(dicts["relative"]),
        "job_title": random.choice(dicts["job_title"]),
        "company": random.choice(dicts["company"]),
        "age": str(random.randint(7, 95)),
        "sex": random.choice(dicts["sex"]),
        "ethnicity": random.choice(dicts["ethnicity"]),
        "sexual_orientation": random.choice(dicts["sexual_orientation"]),
        "school_name": random.choice(dicts["school_name"]),
        "credit_card": gen_credit_card(),
        "username": gen_username(name, surname),
        "secret": gen_secret(),
        "date_of_birth": gen_dob(),
        "date": gen_date(),
    }


def generate_example(domain: str) -> Dict:
    tpl = random.choice(TEMPLATES[domain])
    vals = sample_values(DICTS)
    filled, spans = render_with_spans(tpl, vals)
    noisy = add_noise(filled)
    return {
        "text": noisy,
        "placeholders": tpl,
        "values": vals,
        "entities": spans,
        "meta": {"domain": domain, "noise": noisy != filled},
    }


def main():
    parser = argparse.ArgumentParser(description="Generator syntetycznych danych PII")
    parser.add_argument("--output", default="synthetic.jsonl", help="Ścieżka zapisu")
    parser.add_argument("--count", type=int, default=4000, help="Liczba przykładów do wygenerowania")
    parser.add_argument("--append", action="store_true", help="Dołącz do istniejącego pliku zamiast nadpisywać")
    args = parser.parse_args()

    domains = list(TEMPLATES.keys())
    records: List[str] = []
    if args.append and Path(args.output).exists():
        records = Path(args.output).read_text().splitlines()

    for _ in range(args.count):
        dom = random.choice(domains)
        rec = generate_example(dom)
        records.append(json.dumps(rec, ensure_ascii=False))

    Path(args.output).write_text("\n".join(records))
    print(f"Zapisano {len(records)} przykładów do {args.output}")


if __name__ == "__main__":
    main()
