# Anonymous Microsoft Web Data — klasterovanje

Projekat iz kursa **Istraživanje podataka 2**, Matematički fakultet, Univerzitet u Beogradu.
Tema **36. Anonymous Microsoft Web Data**, metod: **klasterovanje**.

## Zadatak

Nad logovima poseta sajtu `www.microsoft.com` primeniti klasterovanje i grupisati korisnike
prema obrascima posećivanja delova sajta. Zahtevano je preprocesiranje podataka, primena
najmanje pet algoritama klasterovanja, poređenje dobijenih rešenja, prikaz podataka u 2D ili
3D, kao i pravljenje modela sa svim atributima i sa različitim redukovanim skupovima atributa.

## Podaci

Izvor: [UCI Machine Learning Repository, dataset 4](https://archive.ics.uci.edu/dataset/4/anonymous+microsoft+web+data),
DOI [10.24432/C5VS3Q](https://doi.org/10.24432/C5VS3Q), autori Breese, Heckerman i Kadie (1998).
Licenca CC BY 4.0.

Sirovi fajlovi se nalaze u folderu `anonymous+microsoft+web+data/` i **ne menjaju se**:

- `anonymous-msweb.data` — 294 vroot-a i 32711 korisnika sa ukupno 98654 posete
- `anonymous-msweb.test` — dodatnih 5000 korisnika
- `anonymous-msweb.info` — opis formata

Podaci su u retkom ASCII formatu „DST": linije `A` opisuju oblasti sajta (vroot-ove), linije
`C` otvaraju korisnika, a linije `V` nabrajaju vroot-ove koje je taj korisnik posetio.

## Pokretanje

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Instalacija u režimu `-e` znači da je paket `msweb` dostupan svuda u okruženju, pa ni skripte
ni sveske ne moraju da podešavaju `sys.path`.

Provera da okruženje radi:

```bash
python -c "from msweb.config import ROOT, RAW_TRAIN; print(ROOT, RAW_TRAIN.exists())"
```

## Reprodukcija

Skripte se pokreću redom, iz korena projekta:

```bash
python scripts/01_parse.py   # DST -> data/interim/{vroots,users,visits}.csv
pytest                       # provere invarijanti
```

## Struktura projekta

```
anonymous+microsoft+web+data/  sirovi podaci (u repozitorijumu)
data/interim/                  DST preveden u tabele (generiše se)
data/processed/                binarne matrice i skupovi atributa (generiše se)
src/msweb/                     biblioteka projekta
src/msweb/algorithms/          po jedan modul za svaki algoritam klasterovanja
scripts/                       izvršne skripte, pokreću se redom 01, 02, ...
notebooks/                     Jupyter sveske za prikaz rezultata
output/figures/                grafici (generiše se)
output/tables/                 tabele i registar rezultata (generiše se)
output/models/                 konstruisani modeli i oznake klastera (generiše se)
tests/                         provere parsera i sopstvenih implementacija algoritama
report/                        tekstualni deo rada i zapisnik
```

## Literatura i alati

- Charu C. Aggarwal, *Data Mining: The Textbook*, Springer, 2015 (glave 2, 6 i 7)
- Materijali sa kursa: [Istraživanje podataka 2](https://www.istrazivanjepodataka.matf.bg.ac.rs/IstrazivanjePodataka2.html)
- `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`
