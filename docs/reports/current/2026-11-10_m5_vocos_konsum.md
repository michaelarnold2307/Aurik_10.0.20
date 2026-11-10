# M5 — Vocos-Musik-Finetune: Export verifiziert, Konsum verdrahtet (D-K3-90)

**Datum:** 2026-11-10 · **Status:** Verdrahtung fertig, Flag **AUS** · **Klasse:** K3
(Phantom-/Unerreichbarkeits-Pfad) · **Kein Training gestartet, kein Flag-Flip.**

## 1. Ausgangslage

Die Vorsitzung hatte zwei Dinge hinterlassen (Arbeitsbaum schmutzig, kein Commit):

1. **M5-Export (2026-11-09 23:00):** `scripts/export_vocos_48khz_onnx.py --finetune`
   schrieb `models/vocos_48khz/finetune/vocos_48khz_music_ft.onnx`
   (274 969 B Graph + 163 717 120 B Gewichte) plus Export-Manifest
   (`aurik.vocos48k_finetune_export/1`, sha256
   `2dd1c726ff438e69373cf14a2afa1cb31484a9087073e7ad2973150c6d051371`, opset 18).
2. **Ohr-Panel-Autarkie:** `scripts/ohr_panel.py` verblindet und baut die
   Urteils-Session jetzt selbst (`erstelle_trials`/`baue_session`) statt über
   `scripts/ab_harness.py` — Grund: die Harness-Datei wurde von einer parallelen
   Sitzung gleichzeitig verändert (geteilte Repo-Datei = Kollisionsrisiko).
   Determinismus bleibt Seed-basiert (§G5), der Key trägt die Originalpfade.

**Befund dieser Sitzung:** Der Export war für **keinen** Lauf erreichbar.
`plugins/vocos_plugin.py` lud ausschließlich die Basis-Tiers (48 kHz → 44,1 kHz →
24 kHz), der §1.4-Finisher in `backend/core/unified_restorer_v3.py` rief nur
`get_vocos_plugin()`. Ein Artefakt ohne Ladestelle ist genau die D-K3-85-Klasse
(„Modell ohne Pfad") — jede künftige A/B-Messung wäre ein No-Op-Artefakt
geworden, und die Messung hätte das nicht angezeigt.

## 2. Umsetzung

| Ort | Änderung |
| --- | --- |
| `plugins/vocos_plugin.py` | `_MODEL_48K_MUSIC_FT` (Pfad zum Finetune) + **eigener Singleton** `get_vocos_music_ft_plugin()`; Herkunfts-Zeuge `weights_basename` (§III.13 (copilot-instructions.md)) wird beim Laden gesetzt |
| `backend/core/music_model_flags.py` | `use_vocos_music_ft: bool = False` mit Begründung/Domain-Hinweis (Basis ist sprach-trainiert, §III.11/§V6 (copilot-instructions.md)) |
| `backend/core/unified_restorer_v3.py` (§1.4) | wählt das Finetune **nur** bei Flag `True`; der Basis-Aufruf `_vplug = _get_vocos_plg()` bleibt davor stehen, die Auswahl wird geloggt (`Gewichte=…`) |

**Fail-closed-Kette (§V6, §V7):**

- Artefakt fehlt ⇒ **keine** Konstruktion, `logger.warning()` + Basis-Singleton.
- Artefakt vorhanden, aber nicht ladbar (ONNX defekt/Provider fehlt) ⇒ Basis +
  **Negativ-Cache** `_music_ft_unavailable` — der 163-MB-Ladeversuch wird nicht
  je Aufruf wiederholt.
- Flag-Prüfung liegt im Core, nicht im Plugin: das Plugin bekommt keine
  `backend.core`-Abhängigkeit (Layer-Trennung bleibt).
- Default AUS ⇒ der Pfad ist **bit-identisch** zum Bestand (kein zusätzlicher
  Session-Start, kein zusätzlicher Speicher).

## 3. Nachweise

- `tests/unit/test_vocos_music_ft_flag.py` — **8/8 grün**, modellfrei:
  Flag-Default + Quelltext-Pin, Konsumpfad ≡ Exportziel (Erzeuger- und
  Konsumentenname per Quelltext-Pin verbunden), fail-closed ohne Artefakt
  (beweist: kein Ladeversuch), Negativ-Cache (genau **ein** Konstruktionsversuch),
  Singleton, UV3-Auswahl-Pin (Reihenfolge im §1.4-Block).
- Bestand: `tests/unit/test_v99_vocos_plugin.py`,
  `tests/unit/test_export_vocos_48khz_finetune.py`,
  `tests/normative/test_primary_paths_no_fallback.py` — grün (keine Regression).
- `scripts/spec_drift_check.py` — „No spec drift detected" (FILE_REGISTRY-Hash im
  Baseline nachgezogen).

## 4. Ehrlich offen

1. **Wirkung nicht gemessen.** Das Finetune ist erreichbar, aber ob es besser
   klingt als die Basis, ist unbekannt — B3-Never-worsen braucht das freie
   GPU-Fenster (Kette 2 fährt D3-3 FlashSR). Ohne diese Messung **kein** Flip.
2. **C4-Primärrollen-Entscheidung** (Vocos-FT vs. BigVGAN-F3 als
   Studio-Finisher-Stufe 1) bleibt menschlich.
3. Der Panel-Autarkie-Teil ist eine Robustheits-/Kollisionsmaßnahme, keine
   Kalibrierung: die Ohr-Kalibrierung selbst (Hör-Sitzung) ist und bleibt
   menschlich.

## 5. Quer-Audit „Flag ohne Lesestelle“ (D-K3-91)

Weil die M5-Lücke eine Klasse ist (Modell/Artefakt vorhanden, Pfad fehlt), wurde
unmittelbar danach **jeder** Modell-Flag der Flag-Datei auf Lesestellen geprüft
(Suchlauf über `backend/`, `plugins/`, `denker/`, `cli/`, `forensics/`, `scripts/`,
jeweils ohne die Flag-Definition selbst):

| Ergebnis | Flags |
| --- | --- |
| gelesen (21) | `use_df_musik`, `use_sgmse_musik`, `use_mp_senet_musik`, `use_bw_v5`, `use_harmonic_inpainting`, `use_utmos_music`, `use_silero_vad_music`, `use_resemblyzer_music`, `use_scnet_music`, `use_bigvgan_f3_studio`, `use_aspade_declipper`, `use_tf_gridnet_quality`, `use_wf4_warp_head`, `use_tp_v2_phase`, `use_material_crnn`, `use_era_crnn`, `use_defect_crnn`, `use_loudness_dpn`, `use_rt60_regressor`, `use_audibility_budget`, `use_vocos_music_ft` (neu) |
| **nicht gelesen (1)** | **`use_miipher_dit`** |

`use_miipher_dit` (Default **True**) war ein **stiller No-Op**: der MIIPHER-DiT-Block
in `enhance_vocal` lief unbedingt, ein `False` hätte nichts abgeschaltet — der
dokumentierte Per-Modell-Rollback existierte nicht (§V7-Klasse). Behoben ohne
Verhaltensänderung: `_miipher_dit_enabled()` liest den Flag (fail-open `True`,
falls das Flag-Modul nicht importierbar ist), der DIT-Block liegt im `else`-Zweig,
`False` ⇒ `miipher_dit:flag_off` in der `fallback_chain` + INFO-Log. 3 Tests in
`tests/unit/test_sota_vocal_model_router.py` (27 passed).

**Lehre für die nächste Sitzung:** ein Flag, das niemand liest, ist kein
Sicherheitsnetz, sondern eine Behauptung. Der Audit ist billig (ein Suchlauf) und
sollte bei jedem neuen Flag mitlaufen.
