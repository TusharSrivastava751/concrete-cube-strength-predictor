"""Regenerate the original synthetic fixture."""
import csv
from pathlib import Path
import random
rng = random.Random(751)
Path('data').mkdir(exist_ok=True)
with open('data/synthetic_cube_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sample_id', 'batch_id', 'strength_7d_mpa', 'strength_28d_mpa'])
    for batch in range(1, 41):
        early = rng.uniform(14, 38)
        batch_effect = rng.gauss(0, 2.5)
        for specimen in range(1, 3):
            x = round(early + rng.gauss(0, 0.7), 2)
            y = round(7 + 1.23*x + batch_effect + rng.gauss(0, 1.2), 2)
            writer.writerow([f'DEMO-{batch:02d}-{specimen}', f'B{batch:02d}', x, y])
