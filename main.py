"""
Ćw. 1 - Dioda elektroluminescencyjna (LED), Laboratorium Fizyka 3.3.

Opracowanie pomiarów charakterystyk I-V diod LED:
  1) wykresy I-V z niepewnościami u(I), u(U),
  2) potencjał wbudowany V_bi = -b/a z prostej dopasowanej w zakresie dużych napięć,
  3) oporność szeregowa R_s = 1/a,
  4) przerwa energetyczna E_g = hc/lambda,
  5) porównanie z E_g = e*V_bi,
  6) wykresy lnI = f(V) i współczynnik idealności n = q/(k_B T a).

Pliki .tsv: kolumna 1 - napięcie [V], kolumna 2 - prąd [A].
Wyniki: katalog `wykresy/` oraz plik `wyniki.csv`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Stałe fizyczne (SI)
# ----------------------------------------------------------------------------
H = 6.62607015e-34  # J*s
C = 299_792_458.0  # m/s
E = 1.602176634e-19  # C
KB = 1.380649e-23  # J/K
T = 300.0  # K

# ----------------------------------------------------------------------------
# Dokładności multimetrów: u = (proc% * |odczyt| + dgt * rozdzielczość) / sqrt(3)
# UWAGA: wpisz wartości z instrukcji do multimetrów używanych na pracowni.
# ----------------------------------------------------------------------------
MULTIMETR_U = {"proc": 0.5, "dgt": 2, "rozdzielczosc": 1e-4}  # woltomierz [V]
MULTIMETR_I = {"proc": 0.8, "dgt": 3, "rozdzielczosc": 1e-6}  # amperomierz [A]

# ----------------------------------------------------------------------------
# Diody: plik, długość fali [nm], kolor wykresu oraz zakresy dopasowań.
#   frac_duze  - do prostej I(V) brane są punkty z I >= frac_duze * I_max
#   zakres_ln  - zakres prądów [A] liniowej części lnI(V)
# ----------------------------------------------------------------------------
DELTA_LAMBDA_NM = 3.0
DIODY = [
    {"nazwa": "niebieska", "plik": "dioda_niebieska.tsv", "lambda_nm": 460,
     "kolor": "tab:blue", "frac_duze": 0.5, "zakres_ln": (3e-5, 2e-4)},
    {"nazwa": "żółto-zielona", "plik": "dioda_zolto-zielona.tsv", "lambda_nm": 560,
     "kolor": "yellowgreen", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "pomarańczowa", "plik": "dioda_pomaranczowa.tsv", "lambda_nm": 596,
     "kolor": "tab:orange", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "czerwona", "plik": "dioda_czerwona.tsv", "lambda_nm": 650,
     "kolor": "tab:red", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "podczerwona", "plik": "dioda_podczerwona.tsv", "lambda_nm": 970,
     "kolor": "darkred", "frac_duze": 0.5, "zakres_ln": (5e-5, 1e-3)},
]

# na wykresach lnI pomijane są prądy poniżej progu (szum rzędu rozdzielczości amperomierza)
PROG_LN_WYKRES = 5e-6  # A

KATALOG = Path(__file__).resolve().parent
KATALOG_WYKRESOW = KATALOG / "wykresy"


def wczytaj(plik):
    dane = np.loadtxt(KATALOG / plik)
    U, I = dane[:, 0], dane[:, 1]
    kolejnosc = np.argsort(U)
    return U[kolejnosc], I[kolejnosc]


def niepewnosc(odczyt, multimetr):
    dokladnosc = multimetr["proc"] / 100 * np.abs(odczyt) + multimetr["dgt"] * multimetr["rozdzielczosc"]
    return dokladnosc / np.sqrt(3)


def regresja(x, y):
    """Metoda najmniejszych kwadratów y = a*x + b; zwraca a, b, Δa, Δb, R^2."""
    n = len(x)
    xs, ys = x.mean(), y.mean()
    sxx = np.sum((x - xs) ** 2)
    a = np.sum((x - xs) * (y - ys)) / sxx
    b = ys - a * xs
    reszty = y - (a * x + b)
    s2 = np.sum(reszty ** 2) / (n - 2)
    da = np.sqrt(s2 / sxx)
    db = np.sqrt(s2 * (1 / n + xs ** 2 / sxx))
    r2 = 1 - np.sum(reszty ** 2) / np.sum((y - ys) ** 2)
    return a, b, da, db, r2


def analizuj(dioda):
    U, I = wczytaj(dioda["plik"])
    uU = niepewnosc(U, MULTIMETR_U)
    uI = niepewnosc(I, MULTIMETR_I)

    # 2), 3) prosta w zakresie dużych napięć
    maska_duze = I >= dioda["frac_duze"] * I.max()
    a, b, da, db, r2 = regresja(U[maska_duze], I[maska_duze])
    Vbi = -b / a
    uVbi = np.sqrt((b / a ** 2 * da) ** 2 + (db / a) ** 2)
    Rs = 1 / a
    uRs = da / a ** 2

    # 4) E_g z długości fali
    lam = dioda["lambda_nm"] * 1e-9
    u_lam = DELTA_LAMBDA_NM * 1e-9 / np.sqrt(3)
    Eg_lam = H * C / lam / E  # eV
    uEg_lam = H * C / lam ** 2 * u_lam / E

    # 5) E_g = e*V_bi -> w eV liczbowo równe V_bi
    Eg_Vbi, uEg_Vbi = Vbi, uVbi

    # 6) lnI = f(V), współczynnik idealności
    i_min, i_max = dioda["zakres_ln"]
    maska_ln = (I >= i_min) & (I <= i_max)
    a_ln, b_ln, da_ln, db_ln, r2_ln = regresja(U[maska_ln], np.log(I[maska_ln]))
    q_kT = E / (KB * T)
    n_id = q_kT / a_ln
    u_n = q_kT * da_ln / a_ln ** 2
    Is = np.exp(b_ln)

    return {
        "U": U, "I": I, "uU": uU, "uI": uI,
        "maska_duze": maska_duze, "maska_ln": maska_ln,
        "a": a, "b": b, "da": da, "db": db, "r2": r2,
        "Vbi": Vbi, "uVbi": uVbi, "Rs": Rs, "uRs": uRs,
        "Eg_lam": Eg_lam, "uEg_lam": uEg_lam, "Eg_Vbi": Eg_Vbi, "uEg_Vbi": uEg_Vbi,
        "a_ln": a_ln, "b_ln": b_ln, "da_ln": da_ln, "db_ln": db_ln, "r2_ln": r2_ln,
        "n": n_id, "u_n": u_n, "Is": Is,
    }


def wykres_IV(dioda, w):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(w["U"], w["I"] * 1e3, xerr=w["uU"], yerr=w["uI"] * 1e3, fmt="o", ms=2.5,
                color=dioda["kolor"], ecolor="gray", elinewidth=0.6, capsize=1.5, label="pomiar")
    ax.plot(w["U"][w["maska_duze"]], w["I"][w["maska_duze"]] * 1e3, "o", ms=3, mfc="none",
            color="black", mew=0.6, zorder=3, label="punkty do dopasowania")
    x = np.linspace(w["Vbi"], w["U"].max(), 100)
    ax.plot(x, (w["a"] * x + w["b"]) * 1e3, "k--", lw=1.2,
            label=f"I = aU + b\na = ({w['a']*1e3:.2f} ± {w['da']*1e3:.2f}) mA/V\n"
                  f"b = ({w['b']*1e3:.1f} ± {w['db']*1e3:.1f}) mA,  R² = {w['r2']:.4f}")
    ax.axvline(w["Vbi"], color="k", lw=0.6, ls=":")
    ax.annotate(f"$V_{{bi}}$ = ({w['Vbi']:.3f} ± {w['uVbi']:.3f}) V", (w["Vbi"], 0),
                xytext=(10, 25), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title(f"Charakterystyka I-V – dioda {dioda['nazwa']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / f"IV_{Path(dioda['plik']).stem}.png", dpi=200)
    plt.close(fig)


def wykres_lnI(dioda, w):
    widoczne = w["I"] >= PROG_LN_WYKRES
    U, I = w["U"][widoczne], w["I"][widoczne]
    u_lnI = w["uI"][widoczne] / I
    maska = w["maska_ln"][widoczne]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(U, np.log(I), xerr=w["uU"][widoczne], yerr=u_lnI, fmt="o", ms=2.5,
                color=dioda["kolor"], ecolor="gray", elinewidth=0.6, capsize=1.5, label="pomiar")
    ax.plot(U[maska], np.log(I[maska]), "o", ms=3, mfc="none", color="black", mew=0.6,
            zorder=3, label="punkty do dopasowania")
    x = np.linspace(U[maska].min() - 0.1, U[maska].max() + 0.1, 100)
    ax.plot(x, w["a_ln"] * x + w["b_ln"], "k--", lw=1.2,
            label=f"lnI = aU + b\na = ({w['a_ln']:.2f} ± {w['da_ln']:.2f}) 1/V\n"
                  f"b = ({w['b_ln']:.2f} ± {w['db_ln']:.2f}),  R² = {w['r2_ln']:.4f}\n"
                  f"n = {w['n']:.3f} ± {w['u_n']:.3f}")
    ax.set_ylim(np.log(I).min() - 0.5, np.log(I).max() + 1)
    ax.set_xlabel("U [V]")
    ax.set_ylabel("ln(I / 1 A)")
    ax.set_title(f"Charakterystyka lnI = f(U) – dioda {dioda['nazwa']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / f"lnI_{Path(dioda['plik']).stem}.png", dpi=200)
    plt.close(fig)


def wykresy_zbiorcze(wyniki):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dioda, w in wyniki:
        ax.plot(w["U"], w["I"] * 1e3, "o", ms=2, color=dioda["kolor"], label=dioda["nazwa"])
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title("Charakterystyki I-V badanych diod LED")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "IV_wszystkie.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dioda, w in wyniki:
        widoczne = w["I"] >= PROG_LN_WYKRES
        ax.plot(w["U"][widoczne], np.log(w["I"][widoczne]), "o", ms=2, color=dioda["kolor"],
                label=dioda["nazwa"])
    ax.set_xlabel("U [V]")
    ax.set_ylabel("ln(I / 1 A)")
    ax.set_title("Charakterystyki lnI = f(U) badanych diod LED")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "lnI_wszystkie.png", dpi=200)
    plt.close(fig)

    nazwy = [d["nazwa"] for d, _ in wyniki]
    x = np.arange(len(nazwy))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x - 0.1, [w["Eg_lam"] for _, w in wyniki], yerr=[w["uEg_lam"] for _, w in wyniki],
                fmt="o", capsize=4, label=r"$E_g = hc/\lambda$")
    ax.errorbar(x + 0.1, [w["Eg_Vbi"] for _, w in wyniki], yerr=[w["uEg_Vbi"] for _, w in wyniki],
                fmt="s", capsize=4, label=r"$E_g = e\,V_{bi}$")
    ax.set_xticks(x, nazwy)
    ax.set_ylabel(r"$E_g$ [eV]")
    ax.set_title("Porównanie przerw energetycznych")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "Eg_porownanie.png", dpi=200)
    plt.close(fig)


def raport(wyniki):
    print(f"T = {T} K,  q/(k_B T) = {E / (KB * T):.3f} 1/V")
    print(f"Multimetr U: ±({MULTIMETR_U['proc']}% + {MULTIMETR_U['dgt']} dgt), "
          f"rozdz. {MULTIMETR_U['rozdzielczosc']} V")
    print(f"Multimetr I: ±({MULTIMETR_I['proc']}% + {MULTIMETR_I['dgt']} dgt), "
          f"rozdz. {MULTIMETR_I['rozdzielczosc']} A\n")

    for dioda, w in wyniki:
        print(f"=== Dioda {dioda['nazwa']} (λ = {dioda['lambda_nm']} nm) ===")
        print(f"  Prosta I(U), {w['maska_duze'].sum()} pkt (I >= {dioda['frac_duze']:.0%} I_max):")
        print(f"    a = ({w['a']:.5f} ± {w['da']:.5f}) A/V,  b = ({w['b']:.5f} ± {w['db']:.5f}) A,"
              f"  R² = {w['r2']:.5f}")
        print(f"    V_bi = ({w['Vbi']:.4f} ± {w['uVbi']:.4f}) V")
        print(f"    R_s  = ({w['Rs']:.3f} ± {w['uRs']:.3f}) Ω")
        print(f"  E_g = hc/λ   = ({w['Eg_lam']:.4f} ± {w['uEg_lam']:.4f}) eV")
        print(f"  E_g = e·V_bi = ({w['Eg_Vbi']:.4f} ± {w['uEg_Vbi']:.4f}) eV"
              f"  (różnica {w['Eg_Vbi'] - w['Eg_lam']:+.3f} eV, "
              f"{(w['Eg_Vbi'] - w['Eg_lam']) / w['Eg_lam']:+.1%})")
        i_min, i_max = dioda["zakres_ln"]
        print(f"  Prosta lnI(U), {w['maska_ln'].sum()} pkt ({i_min:g} A <= I <= {i_max:g} A):")
        print(f"    a = ({w['a_ln']:.4f} ± {w['da_ln']:.4f}) 1/V,  b = ({w['b_ln']:.4f} ± {w['db_ln']:.4f}),"
              f"  R² = {w['r2_ln']:.5f}")
        print(f"    n = {w['n']:.4f} ± {w['u_n']:.4f},  I_s = exp(b) = {w['Is']:.3e} A\n")

    naglowek = ("dioda;lambda_nm;a_A_V;u_a;b_A;u_b;R2;Vbi_V;u_Vbi;Rs_Ohm;u_Rs;"
                "Eg_lambda_eV;u_Eg_lambda;Eg_Vbi_eV;u_Eg_Vbi;a_ln_1_V;u_a_ln;b_ln;u_b_ln;R2_ln;n;u_n;Is_A")
    wiersze = [naglowek]
    for dioda, w in wyniki:
        wartosci = [w[k] for k in ("a", "da", "b", "db", "r2", "Vbi", "uVbi", "Rs", "uRs",
                                   "Eg_lam", "uEg_lam", "Eg_Vbi", "uEg_Vbi", "a_ln", "da_ln",
                                   "b_ln", "db_ln", "r2_ln", "n", "u_n", "Is")]
        wiersze.append(";".join([dioda["nazwa"], str(dioda["lambda_nm"])] + [f"{v:.6g}" for v in wartosci]))
    (KATALOG / "wyniki.csv").write_text("\n".join(wiersze) + "\n", encoding="utf-8-sig")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    KATALOG_WYKRESOW.mkdir(exist_ok=True)
    wyniki = []
    for dioda in DIODY:
        w = analizuj(dioda)
        wykres_IV(dioda, w)
        wykres_lnI(dioda, w)
        wyniki.append((dioda, w))
    wykresy_zbiorcze(wyniki)
    raport(wyniki)
    print(f"Wykresy zapisano w: {KATALOG_WYKRESOW}")
    print(f"Tabela wyników:     {KATALOG / 'wyniki.csv'}")


if __name__ == "__main__":
    main()
