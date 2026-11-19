# Tutorial: Batch-Verarbeitung

> §15.7: Mehrere Dateien in einem Durchlauf restaurieren.

## Grundlagen

```bash
python -m aurik batch ./input_dir/ --output ./output_dir/
```

## Nach Material gruppieren

```bash
python -m aurik batch ./input_dir/ \
    --output ./output_dir/ \
    --group-by material \
    --mode full
```

## Fortschrittsanzeige

```bash
python -m aurik batch ./input_dir/ \
    --output ./output_dir/ \
    --progress \
    --workers 4
```

[Weitere Details folgen...]
