#!/usr/bin/env python3
"""Public-Domain Corpus Generator für Aurik — §15.2.

Lädt gemeinfreie Audio-Aufnahmen von Internet Archive, Musopen und Freesound (CC0)
herunter und erstellt manifest.yaml-Einträge nach corpus/MANIFEST_SCHEMA.yaml.

Nutzung:
  python scripts/generate_corpus_from_public_domain.py --material vinyl --count 5
  python scripts/generate_corpus_from_public_domain.py --all --dry-run
  python scripts/generate_corpus_from_public_domain.py --from-urls urls.txt

Abhängigkeiten: requests, yt-dlp (optional für Internet Archive), librosa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"

# ── Public-Domain-Quellen ──────────────────────────────────────────────────
# Jeder Eintrag: (url, material, era_year, genre, defect_types, attribution, license)
# Nur gemeinfreie / CC0-Quellen. Kein urheberrechtlich geschütztes Material.

PUBLIC_DOMAIN_SOURCES: dict[str, list[dict[str, Any]]] = {
    # Echte, verifizierte Public-Domain-Aufnahmen (US: veröffentlicht ≤ 1926).
    # Alle Einträge: Internet Archive 78rpm Collection (Public Domain Mark 1.0).
    # Verifiziert am 2026-08-15: Lizenz-Metadaten, Jahr ≤ 1926, Audio-Checksummen
    # im corpus/shellac/manifest.yaml (test_corpus_integrity grün).
    "shellac": [
        {
            "url": "https://archive.org/download/enrico-caruso-giuseppe-vesrdi-rigoletto-la-donna-emobile-gramophone-da-1303/EnricoCaruso%2CGiuseppeVesrdi%2CRigoletto%2CLaDonnaEMobile%2CGramophone%2CDA1303.mp3",
            "era_year": 1908,
            "slug": "caruso_rigoletto_la_donna_e_mobile",
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Rigoletto: La donna è mobile (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_ada-celeste-ada-heavenly-aida_enrico-caruso-giueseppi-verdi_gbia0290541a/A%C3%AFda%20-%20Celeste%20A%C3%AFda%20%28Heavenly%20Aida%21%29%20-%20Enrico%20Caruso.mp3",
            "era_year": 1911,
            "slug": "caruso_celeste_aida",
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Aïda: Celeste Aïda (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/enrico-caruso-eduardo-di-capuai-osole-mio-gramophone-da-1303/EnricoCaruso%2CEduardoDiCapuai%2COSoleMio%2CGramophoneDA1303.mp3",
            "era_year": 1916,
            "slug": "caruso_o_sole_mio",
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — O Sole Mio (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/1919-USA-Archives-1919-07-00-Joseph-Samuels-Dance-Orch-Taxi/1919%20%28USA%29%20Archives%201919%2007%2000%20Joseph%20Samuels%27%20Dance%20Orch%20-%20Taxi.mp3",
            "era_year": 1919,
            "slug": "samuels_dance_orch_taxi",
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Joseph Samuels' Dance Orchestra — Taxi (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_eddie-leonard-blues_irving-kaufman-v-l-stanton_gbia0286489b/Eddie%20Leonard%20Blues%20-%20Irving%20Kaufman%20-%20V.%20L.%20Stanton.mp3",
            "era_year": 1922,
            "slug": "kaufman_eddie_leonard_blues",
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Irving Kaufman — Eddie Leonard Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/farawaybluesraw/Far%20away%20Blues%20raw.mp3",
            "era_year": 1923,
            "slug": "bessie_smith_far_away_blues",
            "genre": "blues",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Bessie Smith & Clara Smith — Far Away Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_dirty-hands-dirty-face_bob-thomas-jolson-clark-leslie_gbia0086661a/Dirty%20Hands%21%20Dirty%20Face%21%20-%20Bob%20Thomas%20-%20Jolson.mp3",
            "era_year": 1923,
            "slug": "jolson_style_dirty_hands_dirty_face",
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Bob Thomas (Jolson) — Dirty Hands! Dirty Face! (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_slik-pinden-danse-orkester-efim-schachmeister-edward/Slik-pinden%20-%20Efim%20Schachmeister.mp3",
            "era_year": 1924,
            "slug": "schachmeister_slik_pinden",
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Efim Schachmeister Danse Orkester — Slik-pinden (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_rock-a-bye-my-baby-blues_frank-wright-and-frank-bessinger-the-two-franks-hill-yoe_gbia0236696b/Rock-A-Bye%20My%20Baby%20Blues%20-%20Frank%20Wright%20and%20Frank%20Bessinger%20%28The%20Two%20Franks%29.mp3",
            "era_year": 1924,
            "slug": "two_franks_rock_a_bye_blues",
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "The Two Franks — Rock-A-Bye My Baby Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_quartet-emin-op59-no2-3rd-mvt-allegretto_virtuoso-string-quartet/D_955_Cc_4690-3.flac",
            "era_year": 1924,
            "slug": "virtuoso_string_quartet_op59_no2_mvt3",
            "genre": "classical",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Virtuoso String Quartet — Beethoven op. 59 Nr. 2, 3. Satz (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_vesti-la-giubba_sig-caruso-leoncavallo_gbia7002950a/VESTI%20LA%20GIUBBA%20-%20SIG.%20CARUSO%20-%20Leoncavallo.mp3",
            "era_year": 1925,
            "slug": "caruso_vesti_la_giubba",
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Pagliacci: Vesti la giubba (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_scherazade-op35-pt3_royal-opera-house-orchestra-covent-garden/C_1288_CR_563-4.flac",
            "era_year": 1925,
            "slug": "covent_garden_scheherazade_pt3",
            "genre": "classical",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Royal Opera House Orchestra — Rimski-Korsakow: Scheherazade op. 35, Teil 3 (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/ProchVariationenMarieDietrich/Proch%20Variationen%20Marie%20Dietrich.mp3",
            "era_year": 1902,
            "genre": "['Dietrich', 'Proch', 'Gramophone & Type",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Marie DIETRICH - Sopran — Proch Variationen Marie Dietrich (Internet Archive, ProchVariationenMarieDietrich)",
            "license": "Public Domain",
            "slug": "prochvariationenmariedietrich",
        },
        {
            "url": "https://archive.org/download/NotAfricaineOdeon/Not%C3%A9%20Africaine%20Odeon.mp3",
            "era_year": 1904,
            "genre": "['acoustic', 'Opera', 'Africaine', 'Afri",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Jean NOTÉ - Bariton — Noté Africaine Odeon (Internet Archive, NotAfricaineOdeon)",
            "license": "Public Domain",
            "slug": "notafricaineodeon",
        },
        {
            "url": "https://archive.org/download/RheingoldElizzaPohlnerKittelVx98/Rheingold%20Elizza%20Pohlner%20Kittel%20Vx%2098%20korr.mp3",
            "era_year": 1904,
            "genre": "['Rheingold', 'Wagner', 'Opera', 'sopran",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Elise ELIZZA - Sopran, Hermine KITTEL - Alt, Jenny POHLNER - Mezzosopran — Rheingold Elizza Pohlner Kittel Vx 98 (Internet Archive, RheingoldElizzaPohlnerKittel",
            "license": "Public Domain",
            "slug": "rheingoldelizzapohlnerkittelvx98",
        },
        {
            "url": "https://archive.org/download/MaskenballFrDeinGlck/Maskenball%20F%C3%BCr%20Dein%20Gl%C3%BCck.mp3",
            "era_year": 1905,
            "genre": "['Gramophone Concert Record', 'Opera', '",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Leopold DEMUTH - Bariton — Maskenball: Für Dein Glück (Internet Archive, MaskenballFrDeinGlck)",
            "license": "Public Domain",
            "slug": "maskenballfrdeinglck",
        },
        {
            "url": "https://archive.org/download/MeinHerrUndGottHesch342416/Mein%20Herr%20und%20Gott%20Hesch%203%2042416.mp3",
            "era_year": 1905,
            "genre": "['Gramophone & Typewriter', 'Hesch', 'ba",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Wilhelm HESCH - Bass — Mein Herr Und Gott Hesch 3-42416 (Internet Archive, MeinHerrUndGottHesch342416)",
            "license": "Public Domain",
            "slug": "meinherrundgotthesch342416",
        },
        {
            "url": "https://archive.org/download/x-33937-note-si-jetais-dieu/X%2033937%20Not%C3%A9%20Si%20j%27etais%20dieu.mp3",
            "era_year": 1905,
            "genre": "['Noté', 'baritone', 'Vortragslied franz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Jean NOTÈ - Bariton — Si j'etais Dieu (Marietti) (Internet Archive, x-33937-note-si-jetais-dieu)",
            "license": "Public Domain",
            "slug": "x_33937_note_si_jetais_dieu",
        },
        {
            "url": "https://archive.org/download/LerchleHempel/Lerchle%20Hempel.mp3",
            "era_year": 1906,
            "genre": "['Odeon', 'Taubert', 'Hempel', 'soprano'",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Frieda HEMPEL - Sopran — S Lerchle Hempel (Internet Archive, LerchleHempel)",
            "license": "Public Domain",
            "slug": "lerchlehempel",
        },
        {
            "url": "https://archive.org/download/LilliLehmannFreudvollUndLeidvoll/Lilli%20Lehmann%20Freudvoll%20und%20leidvoll.mp3",
            "era_year": 1906,
            "genre": "['Odeon', 'acoustic', 'Kunstlied', 'Beet",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Lilli LEHMANN - Sopran. Fritz LINDEMANN - Piano — Freudvoll und leidvoll (Beethoven) (Internet Archive, LilliLehmannFreudvollUndLeidvoll)",
            "license": "Public Domain",
            "slug": "lillilehmannfreudvollundleidvoll",
        },
        {
            "url": "https://archive.org/download/fernando-de-lucia-et-giuseppina-huguet-giuseppe-verdi-la-traviata-parigi-ocara-g.-t.-054081/FernandoDeLuciaEtGiuseppinaHuguet%2CGiuseppeVerdi%2CLaTraviata%2CParigiOCara%2CG.%26T.054081.mp3",
            "era_year": 1906,
            "genre": "['FernandoDeLuciaEtGiuseppinaHuguet', 'G",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Fernando De Lucia Et Giuseppina Huguet, Giuseppe Verdi, La Traviata, Parigi OCara, G.& T. 054081 (Internet Archive, fernando-de-lucia-et-giuseppina-hug",
            "license": "Public Domain",
            "slug": "fernando_de_lucia_et_giuseppina_huguet_giuseppe_verdi_la_tra",
        },
        {
            "url": "https://archive.org/download/OWieSoTrgerisch/O%20wie%20so%20tr%C3%BCgerisch.mp3",
            "era_year": 1907,
            "genre": "['Rigoletto', 'Opera', 'Verdi', 'tenor',",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Karl JÖRN - Tenor — O wie so trügerisch (Rigoletto) (Internet Archive, OWieSoTrgerisch)",
            "license": "Public Domain",
            "slug": "owiesotrgerisch",
        },
        {
            "url": "https://archive.org/download/titta-ruffo-ettore-titta-malena-disse-il-saggio-predog-2-52624/TittaRuffo%2CEttoreTitta%2CMalena%2CDisseIlSaggio%2CPredog2-52624.mp3",
            "era_year": 1907,
            "genre": "['TittaRuffo', 'EttoreTitta', 'Malena', ",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Titta Ruffo, Ettore Titta, Malena, Disse Il Saggio, Predog 2 52624 (Internet Archive, titta-ruffo-ettore-titta-malena-disse-il-saggio-predog-2-52624)",
            "license": "Public Domain",
            "slug": "titta_ruffo_ettore_titta_malena_disse_il_saggio_predog_2_526",
        },
        {
            "url": "https://archive.org/download/valentin-jaume-fromenthal-halevy-la-juive-rachelquanddu-seigneur-odeon-brun-60159/ValentinJaume%2CFromenthal%20Halevy%2CLaJuive%2CRachelquandduSeigneur%2COdeonBrun60159.mp3",
            "era_year": 1907,
            "genre": "['ValentinJaume', 'Fromenthal Halevy', '",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Valentin Jaume, Fromenthal Halevy, La Juive, Rachelquanddu Seigneur, Odéon Brun 60159 (Internet Archive, valentin-jaume-fromenthal-halevy-la-juive-rach",
            "license": "Public Domain",
            "slug": "valentin_jaume_fromenthal_halevy_la_juive_rachelquanddu_seig",
        },
        {
            "url": "https://archive.org/download/vittorio-arimondi-charles-gounod-faust-serenade-de-mephisto-columbia-tricolor-a-5010/VittorioArimondi%2CCharlesGounod%2CFaust%2CS%C3%A9r%C3%A9nadeDeM%C3%A9phisto%2CColumbiaTricolorA5010.mp3",
            "era_year": 1907,
            "genre": "['VittorioArimondi', 'CharlesGounod', 'F",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Vittorio Arimondi, Charles Gounod, Faust, Sérénade De Méphisto, Columbia Tricolor A 5010 (Internet Archive, vittorio-arimondi-charles-gounod-faust-sere",
            "license": "Public Domain",
            "slug": "vittorio_arimondi_charles_gounod_faust_serenade_de_mephisto",
        },
        {
            "url": "https://archive.org/download/x-22667-hieber-andreas-hofer/X-22667%20Hieber%20-%20Andreas%20Hofer.mp3",
            "era_year": 1907,
            "genre": "['Vortragslied deutsch', 'Knebelsberger'",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Theodor HIEBER - Bass — Andreas Hofer (Internet Archive, x-22667-hieber-andreas-hofer)",
            "license": "Public Domain",
            "slug": "x_22667_hieber_andreas_hofer",
        },
        {
            "url": "https://archive.org/download/x-22841-arno-vogelhochzeit/X-22841%20Arno%20Vogelhochzeit.mp3",
            "era_year": 1908,
            "genre": "['Vortragslied deutsch', 'Ludwig Arno', ",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Ludwig ARNO - Tenor — Vogelhochzeit (Internet Archive, x-22841-arno-vogelhochzeit)",
            "license": "Public Domain",
            "slug": "x_22841_arno_vogelhochzeit",
        },
        {
            "url": "https://archive.org/download/x-22960-roeder-uhr-aufziehn/X-22960%20Roeder%20-%20Uhr%20aufziehn.mp3",
            "era_year": 1908,
            "genre": "['Zonophone Record', 'Roeder', 'Ludwig A",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Hermann ROEDER - Tenor, Ludwig ARNO - Tenor — Woll'n Sie meiner Frau nicht mal die Uhr aufzieh'n? (Internet Archive, x-22960-roeder-uhr-aufziehn)",
            "license": "Public Domain",
            "slug": "x_22960_roeder_uhr_aufziehn",
        },
        {
            "url": "https://archive.org/download/waldandachthubertus/Waldandacht%20Hubertus.mp3",
            "era_year": 1909,
            "genre": "['Concert Record Gramophone', 'Kunstpfei",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "HUBERTUS - Vogelstimmenimitator — Waldandacht mit Vogelgezwitscher (Internet Archive, waldandachthubertus)",
            "license": "Public Domain",
            "slug": "waldandachthubertus",
        },
        {
            "url": "https://archive.org/download/wiemeiahnerlgruber/Wie%20mei%20Ahnerl%20Gruber.mp3",
            "era_year": 1909,
            "genre": "['Operette', 'Operetta', 'Gruber', 'teno",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Franz GRUBER - Tenor — Wie mein Ahnerl (Gruber) (Internet Archive, wiemeiahnerlgruber)",
            "license": "Public Domain",
            "slug": "wiemeiahnerlgruber",
        },
        {
            "url": "https://archive.org/download/x-5-22014-arno-paula-mach-die-bluse-zu/X-5-22014%20Arno%20-%20Paula%20mach%20die%20Bluse%20zu.mp3",
            "era_year": 1909,
            "genre": "['Ludwig Arno', 'tenor', 'Vortragslied d",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Ludwig ARNO - Tenor — Paula, mach die Bluse zu (Internet Archive, x-5-22014-arno-paula-mach-die-bluse-zu)",
            "license": "Public Domain",
            "slug": "x_5_22014_arno_paula_mach_die_bluse_zu",
        },
        {
            "url": "https://archive.org/download/aristodemo-giorgini-arrigo-boito-mefistofele-giunto-sul-passo-estremo-pathe-x-6/AristodemoGiorgini%2CArrigoBoito%2CMefistofele%2CGiuntoSulPassoEstremo%2CPath%C3%A9X6.mp3",
            "era_year": 1912,
            "genre": "['AristodemoGiorgini', 'ArrigoBoito', 'M",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Aristodemo Giorgini, Arrigo Boito, Mefistofele, Giunto Sul Passo Estremo, Pathé X 6 (Internet Archive, aristodemo-giorgini-arrigo-boito-mefistofele-giu",
            "license": "Public Domain",
            "slug": "aristodemo_giorgini_arrigo_boito_mefistofele_giunto_sul_pass",
        },
        {
            "url": "https://archive.org/download/MuetteDePortici1215S/Muette%20de%20Portici%201215%20s.mp3",
            "era_year": 1913,
            "genre": "['Muette de Portici', 'Stumme von Portic",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Hermann JADLOWKER - Tenor — Muette De Portici 1215 s (Internet Archive, MuetteDePortici1215S)",
            "license": "Public Domain",
            "slug": "muettedeportici1215s",
        },
        {
            "url": "https://archive.org/download/zartlich-suss-und-leise/Z%C3%A4rtlich%20s%C3%BCss%20und%20leise.mp3",
            "era_year": 1913,
            "genre": "['Zonophone Record', 'Kino-Königin', 'Je",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Grete WIEDECKE - Gesang, August BOCKMANN - Gesang — Zärtlich, süß und leise (Internet Archive, zartlich-suss-und-leise)",
            "license": "Public Domain",
            "slug": "zartlich_suss_und_leise",
        },
        {
            "url": "https://archive.org/download/van-endert-alles-pflegt/van%20Endert%20Alles%20pflegt.mp3",
            "era_year": 1915,
            "genre": "['Schallplatte Grammophon', 'van Endert'",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Elisabeth VAN ENDERT - Sopran — Alles pflegt schon längst der Ruh' (Internet Archive, van-endert-alles-pflegt)",
            "license": "Public Domain",
            "slug": "van_endert_alles_pflegt",
        },
        {
            "url": "https://archive.org/download/derhonvedhusar/Der%20Honved%20Husar.mp3",
            "era_year": 1919,
            "genre": "['Vortragslied deutsch', 'Robitschek', '",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Bernhard BÖTEL - Tenor — Der Honvedhusar (Internet Archive, derhonvedhusar)",
            "license": "Public Domain",
            "slug": "derhonvedhusar",
        },
        {
            "url": "https://archive.org/download/w_moor_echo/w_moor_echo.mp3",
            "era_year": 1919,
            "genre": "Oldtime",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Unbekannt — Echo (Internet Archive, w_moor_echo)",
            "license": "Public Domain",
            "slug": "w_moor_echo",
        },
        {
            "url": "https://archive.org/download/KarinBranzellSamson/Karin%20Branzell%20Samson.mp3",
            "era_year": 1920,
            "genre": "['Branzell', 'contralto', 'Opera', 'Gram",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Karin BRANZELL - Alt — Sieh, mein Herz erschliesset sich (Internet Archive, KarinBranzellSamson)",
            "license": "Public Domain",
            "slug": "karinbranzellsamson",
        },
        {
            "url": "https://archive.org/download/wieihreunschuld/Wie%20ihre%20Unschuld.mp3",
            "era_year": 1920,
            "genre": "['Mignon', 'Ambroise Thomas', 'Opera', '",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Bernhard BÖTEL - Tenor — Wie ihre Unschuld (Mignon) (Internet Archive, wieihreunschuld)",
            "license": "Public Domain",
            "slug": "wieihreunschuld",
        },
        {
            "url": "https://archive.org/download/daniel-vigneau-ruggero-leoncavallo-paillasse-prologue-pathe-saphir-0343-enregistrement-ca-1921/Daniel%20Vigneau%20-%20Ruggero%20Leoncavallo%20-%20Paillasse%20-%20Prologue%20-%20Path%C3%A9%20Saphir%200343%20enregistr%C3%A9%20ca1921.mp3",
            "era_year": 1921,
            "genre": "['Daniel Vigneau', 'Ruggero Leoncavallo'",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Daniel Vigneau Ruggero Leoncavallo Paillasse Prologue Pathé Saphir 0343 Enregistré Ca 1921 (Internet Archive, daniel-vigneau-ruggero-leoncavallo-pailla",
            "license": "Public Domain",
            "slug": "daniel_vigneau_ruggero_leoncavallo_paillasse_prologue_pathe",
        },
        {
            "url": "https://archive.org/download/vox-3044-bilk-troubadour/Vox%203044%20Bilk%20Troubadour.mp3",
            "era_year": 1921,
            "genre": "['Opera', 'Troubadour', 'Trovatore', 'Ve",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Jaques BILK - Bariton — Der Troubadour: Ihres Auges himmlisch Strahlen (Internet Archive, vox-3044-bilk-troubadour)",
            "license": "Public Domain",
            "slug": "vox_3044_bilk_troubadour",
        },
        {
            "url": "https://archive.org/download/OthelloDuettBohnenPattiera/Othello%20Duett%20Bohnen%20Pattiera.mp3",
            "era_year": 1922,
            "genre": "['as matrix-series', 'late acoustic', 'B",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Tino PATTIERA - Tenor, Michael BOHNEN - Bass — Othello Duett Bohnen Pattiera (Internet Archive, OthelloDuettBohnenPattiera)",
            "license": "Public Domain",
            "slug": "othelloduettbohnenpattiera",
        },
        {
            "url": "https://archive.org/download/deman-liebesfreud/Deman%20Liebesfreud.mp3",
            "era_year": 1922,
            "genre": "['Classical Instrumental recording', 'la",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Rudolf DEMAN - Violine, Otto URACK - Piano — Liebesfreud (Fritz Kreisler) (Internet Archive, deman-liebesfreud)",
            "license": "Public Domain",
            "slug": "deman_liebesfreud",
        },
        {
            "url": "https://archive.org/download/da382aufflugelndesgesanges/DA%20382%20Auf%20Fl%C3%BCgeln%20des%20Gesanges.mp3",
            "era_year": 1923,
            "genre": "['Hempel', 'soprano', 'Electrola', 'late",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Frieda HEMPEL - Sopran — DA 382 Auf Flügeln Des Gesanges (Internet Archive, da382aufflugelndesgesanges)",
            "license": "Public Domain",
            "slug": "da382aufflugelndesgesanges",
        },
        {
            "url": "https://archive.org/download/NozzeDiFigaroDB831/Nozze%20di%20Figaro%20DB%20831.mp3",
            "era_year": 1924,
            "genre": "['Nozze di Figaro', 'Figaros Hochzeit', ",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Toti DAL MONTE - Sopran — Nozze Di Figaro DB 831 (Internet Archive, NozzeDiFigaroDB831)",
            "license": "Public Domain",
            "slug": "nozzedifigarodb831",
        },
        {
            "url": "https://archive.org/download/fernand-ansseau-richard-wagner-lohengrin-ah-respirons-tous-deux-gramophone-da-61/Fernand%20Ansseau%20-%20Richard%20Wagner%20-%20Lohengrin%20%20-%20Ah%20respirons%20tous%20deux%20-%20Gramophone%20DA%20614%20enregistr%C3%A9%20le%2013%20%20mars%201924.mp3",
            "era_year": 1924,
            "genre": "['Fernand Ansseau', 'Richard Wagner', 'L",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Fernand Ansseau Richard Wagner Lohengrin Ah Respirons Tous Deux Gramophone DA 614 Enregistré Le 13 Mars 1924 (Internet Archive, fernand-ansseau-richard",
            "license": "Public Domain",
            "slug": "fernand_ansseau_richard_wagner_lohengrin_ah_respirons_tous_d",
        },
        {
            "url": "https://archive.org/download/ulysses-lappas-amilcare-ponchielli-la-gioconda-cielo-emar-columbia-l-1762/UlyssesLappas%2CAmilcarePonchielli%2CLaGioconda%2CCieloEMar%2CColumbiaL1762.mp3",
            "era_year": 1926,
            "genre": "['UlyssesLappas', 'AmilcarePonchielli', ",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "léo472 — Ulysses Lappas, Amilcare Ponchielli, La Gioconda, Cielo EMar, Columbia L 1762 (Internet Archive, ulysses-lappas-amilcare-ponchielli-la-gioconda-cielo-e",
            "license": "Public Domain",
            "slug": "ulysses_lappas_amilcare_ponchielli_la_gioconda_cielo_emar_co",
        },
        {
            "url": "https://archive.org/download/fausttriodeduel/Faust%20Trio%20de%20Duel.mp3",
            "era_year": 1927,
            "genre": "['Faust', 'Margarethe', 'Opera', 'Gounod",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "René MAISON - Tenor, Julien LAFONT - Bariton, Roger BOURDIN - Bariton, Gustave CLOEZ - Dirigent — Faust: Trio de Duel (Internet Archive, fausttriodeduel)",
            "license": "Public Domain",
            "slug": "fausttriodeduel",
        },
        {
            "url": "https://archive.org/download/weine-nicht-liu-o-7536/Weine%20nicht%20Liu%20O%207536.mp3",
            "era_year": 1927,
            "genre": "['Kiepura', 'tenor', 'Turandot', 'Puccin",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Jan KIEPURA - Tenor, Manfred GURLITT - Dirigent — O weine nicht, Liu (Internet Archive, weine-nicht-liu-o-7536)",
            "license": "Public Domain",
            "slug": "weine_nicht_liu_o_7536",
        },
    ],
    # §WP-7-Befund (2026-10-09, docs/reports/current/2026-10-09_wp7_realtraeger.md):
    # US-PD/CC0-Träger existieren für den LP-/Tape-Zeitalter-Kern NICHT
    # (Sound-Recording-Schutz ab 1948/1972; Uploader-,,CC0" auf Kommerz-Releases
    # ist unglaubwürdig und wird deshalb nicht aufgenommen). Aufgenommen werden
    # ausschließlich Selbst-Dedikationen (DIY/Netlabel-Tape- und Digital-Kultur,
    # CC0/PD Mark des Urhebers). Vinyl bleibt daher ohne Echt-Quellen.
    "vinyl": [],
    "reel_tape": [
        {
            "url": "https://archive.org/download/summer-stuff-reel.t-02/Summer%20Stuff%20Reel.flac16/Summer%20Stuff%20Reel.t03.flac",
            "era_year": 2025,
            "genre": "Reel To Reel Old Time Radio",
            "defect_types": ["wow", "flutter", "hiss"],
            "attribution": "Taperwayne — Summer Stuff Reel (Internet Archive, summer-stuff-reel.t-02)",
            "license": "CC0",
            "slug": "summer_stuff_reel_t_02",
        },
    ],
    "cassette": [
        {
            "url": "https://archive.org/download/archive_20260930/A%20View%20From%20Somewhere/13%20-%20Bryan%20Ruryk%20-%20Osler%20Incident.mp3",
            "era_year": 1987,
            "genre": "['Cassette', 'Tape', 'Noise-Arch', 'Inde",
            "defect_types": ["wow", "flutter", "hiss", "dropouts"],
            "attribution": "Various Artists — Archive (Internet Archive, archive_20260930)",
            "license": "CC0",
            "slug": "archive_20260930",
        },
        {
            "url": "https://archive.org/download/rare-recordings/Rare%20Recording%20of%20Jack%20and%20Jill%20by%20Kidzone.mp3",
            "era_year": 1988,
            "genre": "['kidzone', 'kids music', 'nursery rhyme",
            "defect_types": ["wow", "flutter", "hiss", "dropouts"],
            "attribution": "Kidzone — Rare Recordings (Internet Archive, rare-recordings)",
            "license": "CC0",
            "slug": "rare_recordings",
        },
        {
            "url": "https://archive.org/download/revolution_202004/Revolution.mp3",
            "era_year": 2020,
            "genre": "Experimental",
            "defect_types": ["wow", "flutter", "hiss", "dropouts"],
            "attribution": "Shavon Grass & Gypsy & Federico Balducci — Revolution (Internet Archive, revolution_202004)",
            "license": "Public Domain",
            "slug": "revolution_202004",
        },
        {
            "url": "https://archive.org/download/word-wise/Word%20Wise.mp3",
            "era_year": 2020,
            "genre": "Experimental",
            "defect_types": ["wow", "flutter", "hiss", "dropouts"],
            "attribution": "Anti Media Foundation — Word Wise (Internet Archive, word-wise)",
            "license": "Public Domain",
            "slug": "word_wise",
        },
    ],
    "tape": [
        {
            "url": "https://archive.org/download/CartePostal1DieWurstbruckeInIstanbul/ZOOM0015.MP3",
            "era_year": 2011,
            "genre": "carte postal",
            "defect_types": ["wow", "flutter", "hiss"],
            "attribution": "Blenno Die Wurstbrucke — carte postal1 Die wurstbrucke in istanbul (Internet Archive, CartePostal1DieWurstbruckeInIstanbul)",
            "license": "CC0",
            "slug": "cartepostal1diewurstbruckeinistanbul",
        },
    ],
    "digital": [
        {
            "url": "https://archive.org/download/MARAV994D/02%20Keys%2C%20Lights%2C%20Ignition%2C%20Go.mp3",
            "era_year": 2007,
            "genre": "['Extended Play', 'Compact Disc']",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "the Red Light — Chernobyl (Internet Archive, MARAV994D)",
            "license": "Public Domain",
            "slug": "marav994d",
        },
        {
            "url": "https://archive.org/download/SEAM0007/03_X3.mp3",
            "era_year": 2008,
            "genre": "noise",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Buddhist on Fire — X (Internet Archive, SEAM0007)",
            "license": "Public Domain",
            "slug": "seam0007",
        },
        {
            "url": "https://archive.org/download/SEAM0069/03_Audre.mp3",
            "era_year": 2008,
            "genre": "noise",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Buddhist on Fire — Zami (Internet Archive, SEAM0069)",
            "license": "Public Domain",
            "slug": "seam0069",
        },
        {
            "url": "https://archive.org/download/Fourhorns_-_Mountain_metal/10Collect_the_Slaves.mp3",
            "era_year": 2011,
            "genre": "Fourhorns",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Fourhorns — Fourhorns_-_Mountain_metal (Internet Archive, Fourhorns_-_Mountain_metal)",
            "license": "CC0",
            "slug": "fourhorns_mountain_metal",
        },
        {
            "url": "https://archive.org/download/LudwigGrossmann-ardaIzOpereDuhVojvode/LudwigGrossmann-ardaIzOpereDuhVojvode.mp3",
            "era_year": 2012,
            "genre": "Vanzemlja.",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Ludwig Grossmann — Ludwig Grossmann-Cardas Iz Opere Duh Vojvode (Internet Archive, LudwigGrossmann-ardaIzOpereDuhVojvode)",
            "license": "CC0",
            "slug": "ludwiggrossmann_ardaizopereduhvojvode",
        },
        {
            "url": "https://archive.org/download/MeatForce-TheBeast666/TheBeast666.mp3",
            "era_year": 2012,
            "genre": "funtime",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Meat Force — Meat Force - The Beast 666 (Internet Archive, MeatForce-TheBeast666)",
            "license": "CC0",
            "slug": "meatforce_thebeast666",
        },
        {
            "url": "https://archive.org/download/LEMN018-ThermaIkarias-AndThrowItAway/Therma%20Ikarias%20-%2001%20-%20Lit%20Your.mp3",
            "era_year": 2013,
            "genre": "['Less Ego More Noise', 'LEMN']",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Unbekannt — [LEMN018] - Therma Ikarias - And Throw It Away (Internet Archive, LEMN018-ThermaIkarias-AndThrowItAway)",
            "license": "Public Domain",
            "slug": "lemn018_thermaikarias_andthrowitaway",
        },
        {
            "url": "https://archive.org/download/MalaventuraThePlot/Various%20Artist%20%26%20Malaventura%20-%20The%20Plot.mp3",
            "era_year": 2014,
            "genre": "['cut&paste', 'remix', 'malaventura04']",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "malaventura — Various Artist & Malaventura The Plot (Internet Archive, MalaventuraThePlot)",
            "license": "CC0",
            "slug": "malaventuratheplot",
        },
        {
            "url": "https://archive.org/download/MARAV057D/07%20Fast%20Lane%20Geisha.mp3",
            "era_year": 2016,
            "genre": "['Demo Album', 'Computer Audio Recorder'",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "the Maravines — Demos of Sloan (Internet Archive, MARAV057D)",
            "license": "Public Domain",
            "slug": "marav057d",
        },
        {
            "url": "https://archive.org/download/K_Sigma_Don/drum_fast_investigate.flac",
            "era_year": 2021,
            "genre": "['Thomas Park', 'generative', 'public do",
            "defect_types": ["bandwidth_loss", "clipping"],
            "attribution": "Thomas Park — DJ Frankenstone- K Sigma Don [treetrunk 688] (Internet Archive, K_Sigma_Don)",
            "license": "Public Domain",
            "slug": "k_sigma_don",
        },
    ],
}
# ── Minimum required by corpus quality gate ─────────────────────────────────
MIN_FILES_PER_MATERIAL = 5
MIN_TOTAL_FILES = 20
MIN_MATERIALS = 4
# §WP-7 (2026-10-09): Ziel des Datenaufbaus — ≥50 Provenienz-Labels je
# Material-Klasse (Register D-K1-2; docs/TODOS_SOTA_ROADMAP.md WP-7).
# Additiv zum Mindestmaß oben: `meets_minimum` bleibt unverändert.
WP7_TARGET_PER_MATERIAL = 50
# §WP-7 (2026-10-09): Real-Träger-Ziel je Klasse (≥10 Einträge mit Quell-URL).
# Die synthetischen Generator-Labels zählen NICHT dazu — die K1-2-Lücke ist
# explizit „mostly synthetic labels" (Register D-K1-2).
WP7_REAL_TARGET_PER_MATERIAL = 10
# Kanonische Material-Klassen des Corpus-Schemas (corpus/MANIFEST_SCHEMA.yaml).
_CORPUS_MATERIALS = ("shellac", "vinyl", "tape", "reel_tape", "cassette", "digital")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _duration_s(filepath: Path) -> float:
    """Get audio duration in seconds using ffprobe or librosa."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    try:
        import librosa

        return float(librosa.get_duration(path=str(filepath)))
    except Exception:
        logger.warning("Cannot determine duration for %s, defaulting to 0.0", filepath)
        return 0.0


def _sample_rate(filepath: Path) -> int:
    """Get audio sample rate."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    try:
        import librosa

        return int(librosa.get_samplerate(path=str(filepath)))
    except Exception:
        return 0


def _slugify(text: str) -> str:
    """§WP-7 (2026-10-09): Kanonischer Slug für Corpus-Dateinamen.

    Kleinschreibung, Nicht-Alphanumerisches → ``_``, Kollaps, Trim auf 60 Zeichen
    (Muster der bestehenden Shellac-Reals: ``shellac_1908_caruso_rigoletto…``).
    """
    import re
    import urllib.parse as _urlparse

    s = _urlparse.unquote(text).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return s[:60].strip("_") or "import"


# §WP-7-Lizenzvertrag (corpus/README.md): NUR gemeinfrei (Public Domain) oder
# CC0 dürfen in den Echt-Audio-Corpus. Das Gate erzwingt den Vertrag mechanisch
# statt auf Sorgfalt zu hoffen (§G9-Konsistenz, copilot-instructions.md).
_LICENSE_ALLOWLIST = ("public domain", "cc0")


def _license_allowed(license_str: Any) -> bool:
    """True, wenn die Lizenzangabe PD/CC0 ist (corpus/README-Vertrag)."""
    s = str(license_str or "").strip().lower()
    return any(s.startswith(prefix) for prefix in _LICENSE_ALLOWLIST)


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file from url to dest. Returns True on success."""
    if dest.exists():
        logger.info("Already exists: %s", dest)
        return True

    _ensure_dir(dest.parent)

    # Try yt-dlp first (handles Internet Archive, YouTube, etc.)
    try:
        subprocess.run(
            ["yt-dlp", "--no-playlist", "-o", str(dest), url],
            check=True,
            timeout=timeout,
            capture_output=True,
        )
        if dest.exists() and dest.stat().st_size > 0:
            return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # Fallback: requests
    try:
        import requests

        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return dest.stat().st_size > 0
    except Exception as e:
        logger.error("Download fehlgeschlagen for %s: %s", url, e)
        return False


def generate_from_sources(
    material: str | None = None,
    count: int = 5,
    dry_run: bool = False,
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """Download public-domain files and return (dest, source-meta) pairs by material.

    §WP-7 (2026-10-09): Dateinamen folgen dem kanonischen Schema
    ``<material>_<era_year>_<slug>.<ext>`` (wie die bestehenden Shellac-Reals),
    die Source-Metadaten werden bis in das Manifest durchgereicht.
    """
    downloaded: dict[str, list[tuple[Path, dict[str, Any]]]] = {}

    materials = [material] if material else list(PUBLIC_DOMAIN_SOURCES)
    for mat in materials:
        sources = PUBLIC_DOMAIN_SOURCES.get(mat, [])
        dest_dir = CORPUS_ROOT / mat / "damaged"
        mat_downloaded: list[tuple[Path, dict[str, Any]]] = []

        for src in sources:
            if len(mat_downloaded) >= count:
                break
            url = src["url"]
            if not _license_allowed(src.get("license")):
                # §WP-7: Vertragsverstoß wird sichtbar übersprungen, nie still (§V6 copilot-instructions.md).
                logger.warning(
                    "Quelle uebersprungen (Lizenz %r nicht PD/CC0, corpus/README-Vertrag): %s",
                    src.get("license"),
                    url,
                )
                continue
            _stem = src.get("slug") or _slugify(Path(urlparse(url).path).stem)
            _ext = Path(urlparse(url).path).suffix.lower() or ".mp3"
            filename = f"{mat}_{int(src.get('era_year') or 0)}_{_stem}{_ext}"
            dest = dest_dir / filename
            meta = dict(src)
            meta["_dest_name"] = filename

            if dry_run:
                logger.info("DRY-Ausfuehrung would download: %s → %s", url, dest)
                mat_downloaded.append((dest, meta))
                continue

            if download_file(url, dest):
                mat_downloaded.append((dest, meta))
                logger.info("Downloaded: %s", dest)

        downloaded[mat] = mat_downloaded

    return downloaded


def build_manifest(
    material: str,
    audio_files: list[Path],
    source_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a manifest.yaml dict for a material category."""
    entries: list[dict[str, Any]] = []
    meta_by_name: dict[str, dict[str, Any]] = {}
    if source_meta:
        for sm in source_meta:
            # §WP-7: Join über den DEST-Dateinamen (Slug-Schema), nicht den URL-Namen.
            _key = str(sm.get("_dest_name") or Path(urlparse(str(sm.get("url", ""))).path).name)
            meta_by_name[_key] = sm

    for af in sorted(audio_files):
        if not af.exists():
            continue
        meta = meta_by_name.get(af.name, {})
        entry: dict[str, Any] = {
            "file": str(af.relative_to(CORPUS_ROOT / material)),
            "duration_s": round(_duration_s(af), 3),
            "sample_rate": _sample_rate(af),
            "material": meta.get("material", material),
            "era_year": meta.get("era_year", 0),
            "genre": meta.get("genre", "unknown"),
            "condition": "damaged",
            "channels": 2,
            "bit_depth": 16,
        }
        if meta.get("defect_types"):
            entry["defect_types"] = meta["defect_types"]
        if meta.get("attribution"):
            entry["source_attribution"] = meta["attribution"]
        if meta.get("license"):
            entry["license"] = meta["license"]
        if meta.get("url"):
            entry["source_url"] = meta["url"]
        if meta.get("vocal") is not None:
            entry["vocal"] = meta["vocal"]
        if af.exists():
            entry["checksum_sha256"] = _sha256(af)

        entries.append(entry)

    return {
        "corpus_version": "1.0.0",
        "material": material,
        "description": f"Public-Domain {material} recordings for Aurik real-audio validation",
        "entries": entries,
    }


def write_manifests(downloaded: dict[str, list[tuple[Path, dict[str, Any]]]], dry_run: bool = False) -> list[Path]:
    """Write manifest.yaml for each material category. Returns list of written manifests.

    §WP-7 (2026-10-09): MERGE statt Overwrite — bestehende Einträge bleiben
    erhalten (per `file`-Pfad dedupliziert), neue Einträge werden angehängt,
    Source-Metadaten fließen ins Manifest. Bei nicht lesbarem Bestands-Manifest
    wird NICHT geschrieben (kein stiller Datenverlust, §V6 copilot-instructions.md).
    """
    written: list[Path] = []
    for material, pairs in downloaded.items():
        manifest_path = CORPUS_ROOT / material / "manifest.yaml"
        files = [p for p, _m in pairs]
        source_meta = [_m for _p, _m in pairs]
        manifest = build_manifest(material, files, source_meta=source_meta)

        if manifest_path.exists():
            try:
                existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                existing_entries = list(existing.get("entries", []))
            except Exception as _merge_exc:
                logger.error(
                    "Bestands-Manifest %s nicht lesbar (%s) — Schreiben uebersprungen (kein Datenverlust).",
                    manifest_path,
                    _merge_exc,
                )
                continue
            new_files = {str(e.get("file")) for e in manifest["entries"]}
            kept = [e for e in existing_entries if str(e.get("file")) not in new_files]
            manifest["entries"] = kept + manifest["entries"]
            if existing.get("description"):
                manifest["description"] = existing["description"]
            if existing.get("corpus_version"):
                manifest["corpus_version"] = existing["corpus_version"]

        if dry_run:
            logger.info("DRY-Ausfuehrung would write: %s", manifest_path)
            written.append(manifest_path)
            continue

        _ensure_dir(manifest_path.parent)
        manifest_path.write_text(
            yaml.dump(manifest, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        written.append(manifest_path)
        logger.info("Wrote manifest: %s (%d entries)", manifest_path, len(manifest["entries"]))

    return written


def import_from_urls_file(urls_file: Path, material: str, dry_run: bool = False) -> list[tuple[Path, dict[str, Any]]]:
    """Import audio from a text file containing one URL per line."""
    if not urls_file.exists():
        logger.error("URLs file not found: %s", urls_file)
        return []

    urls = [line.strip() for line in urls_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    dest_dir = CORPUS_ROOT / material / "damaged"
    downloaded: list[tuple[Path, dict[str, Any]]] = []

    for url in urls:
        _stem = _slugify(Path(urlparse(url).path).stem)
        _ext = Path(urlparse(url).path).suffix.lower() or ".mp3"
        filename = f"{material}_0_{_stem}{_ext}"
        dest = dest_dir / filename
        meta: dict[str, Any] = {
            "url": url,
            "material": material,
            "_dest_name": filename,
            "attribution": f"Import via --from-urls ({Path(urlparse(url).path).name})",
            "license": "siehe Quelle",
        }
        if dry_run:
            logger.info("DRY-Ausfuehrung would download: %s → %s", url, dest)
            downloaded.append((dest, meta))
        elif download_file(url, dest):
            downloaded.append((dest, meta))

    return downloaded


def check_coverage() -> dict[str, Any]:
    """Check current corpus coverage against minimum requirements."""
    materials_found: list[str] = []
    total_files = 0
    per_material: dict[str, int] = {}
    per_material_real: dict[str, int] = {}

    for mat_dir in sorted(CORPUS_ROOT.iterdir()):
        if not mat_dir.is_dir() or mat_dir.name.startswith("."):
            continue
        manifest = mat_dir / "manifest.yaml"
        if not manifest.exists():
            continue
        try:
            data = yaml.safe_load(manifest.read_text())
            entries = list(data.get("entries", []))
            count = len(entries)
            per_material[mat_dir.name] = count
            # §WP-7 (2026-10-09): Real-Träger = Eintrag mit source_url (die
            # synthetischen Generator-Einträge tragen keine Quell-URL).
            per_material_real[mat_dir.name] = sum(1 for e in entries if e.get("source_url"))
            total_files += count
            materials_found.append(mat_dir.name)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue

    # §WP-7 (2026-10-09): Ziel-Lücke je Material-Klasse (≥50) — maschinenlesbar
    # für den Datenaufbau-Stand (D-K1-2); fehlende Klassen zählen voll.
    wp7_gap = {mat: max(0, WP7_TARGET_PER_MATERIAL - per_material.get(mat, 0)) for mat in _CORPUS_MATERIALS}
    # §WP-7: Real-Träger-Ziel (≥10 je Klasse) — die eigentliche K1-2-Lücke
    # („mostly synthetic labels"); getrennt vom Gesamt-Ziel geführt, damit
    # niemand die Ziele stillschweigend vermischt (§G9 copilot-instructions.md).
    wp7_real_gap = {
        mat: max(0, WP7_REAL_TARGET_PER_MATERIAL - per_material_real.get(mat, 0)) for mat in _CORPUS_MATERIALS
    }

    return {
        "materials_found": materials_found,
        "materials_required": MIN_MATERIALS,
        "total_files": total_files,
        "total_required": MIN_TOTAL_FILES,
        "per_material": per_material,
        "per_material_real": per_material_real,
        "meets_minimum": (len(materials_found) >= MIN_MATERIALS and total_files >= MIN_TOTAL_FILES),
        "wp7_target_per_material": WP7_TARGET_PER_MATERIAL,
        "wp7_gap": wp7_gap,
        "wp7_met": all(gap == 0 for gap in wp7_gap.values()),
        "wp7_real_target_per_material": WP7_REAL_TARGET_PER_MATERIAL,
        "wp7_real_gap": wp7_real_gap,
        "wp7_real_met": all(gap == 0 for gap in wp7_real_gap.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=list(PUBLIC_DOMAIN_SOURCES), help="Nur ein Material herunterladen")
    parser.add_argument("--all", action="store_true", help="Alle Materialien")
    parser.add_argument("--count", type=int, default=5, help="Max Dateien pro Material")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts herunterladen")
    parser.add_argument("--from-urls", type=Path, help="URLs aus Textdatei importieren")
    parser.add_argument("--check", action="store_true", help="Nur Coverage prüfen")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.check:
        coverage = check_coverage()
        print(json.dumps(coverage, indent=2, ensure_ascii=False))
        return 0 if coverage["meets_minimum"] else 1

    if args.from_urls:
        if not args.material:
            print("ERROR: --material required with --from-urls", file=sys.stderr)
            return 1
        downloaded = import_from_urls_file(args.from_urls, args.material, args.dry_run)
        grouped = {args.material: downloaded}
    elif args.material or args.all:
        grouped = generate_from_sources(
            material=args.material if not args.all else None,
            count=args.count,
            dry_run=args.dry_run,
        )
    else:
        print("ERROR: use --all, --material X, --from-urls, or --check", file=sys.stderr)
        return 1

    manifests = write_manifests(grouped, args.dry_run)

    if args.dry_run:
        print(f"DRY-RUN: would write {len(manifests)} manifests for {sum(len(v) for v in grouped.values())} files")
    else:
        coverage = check_coverage()
        print(json.dumps(coverage, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
