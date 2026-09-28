"""
デモ用の UST を生成する（すべてパブリックドメインの旋律）。

    kaeru    かえるのうた（ドイツ民謡）
    kirakira きらきら星（フランス民謡）
    chouchou ちょうちょ（ドイツ民謡）
    scale    ドレミの音階に「あいうえお」を乗せたもの（母音の違いを見せる用）

使い方: python engine/make_songs.py
"""

from pathlib import Path

SONGS = Path(__file__).resolve().parent.parent / "songs"

Q = 480          # 4分音符のtick
H, E = Q * 2, Q // 2

DEFS = {
    "kaeru": (110, "かえるのうた", [
        ("か", 60, Q), ("え", 62, Q), ("る", 64, Q), ("の", 65, Q),
        ("う", 64, Q), ("た", 62, Q), ("が", 60, H), ("R", 0, Q),
        ("き", 60, Q), ("こ", 62, Q), ("え", 64, Q), ("て", 65, Q),
        ("く", 64, Q), ("る", 62, Q), ("よ", 60, H), ("R", 0, Q),
        ("け", 60, E), ("ろ", 60, E), ("け", 60, E), ("ろ", 60, E),
        ("け", 60, E), ("ろ", 60, E), ("け", 60, E), ("ろ", 60, E),
        ("く", 60, Q), ("わ", 62, Q), ("く", 64, Q), ("わ", 65, Q),
        ("け", 64, Q), ("ろ", 62, Q), ("け", 60, H), ("R", 0, Q),
    ]),
    "kirakira": (100, "きらきら星", [
        ("き", 60, Q), ("ら", 60, Q), ("き", 67, Q), ("ら", 67, Q),
        ("ひ", 69, Q), ("か", 69, Q), ("る", 67, H),
        ("お", 65, Q), ("そ", 65, Q), ("ら", 64, Q), ("の", 64, Q),
        ("ほ", 62, Q), ("し", 62, Q), ("よ", 60, H), ("R", 0, Q),
        ("み", 67, Q), ("な", 67, Q), ("ひ", 65, Q), ("か", 65, Q),
        ("る", 64, Q), ("よ", 64, Q), ("R", 62, H),
    ]),
    "chouchou": (108, "ちょうちょ", [
        ("ちょ", 67, Q), ("う", 64, Q), ("ちょ", 64, H),
        ("ちょ", 65, Q), ("う", 62, Q), ("ちょ", 62, H),
        ("な", 60, Q), ("の", 62, Q), ("は", 64, Q), ("に", 65, Q),
        ("と", 67, Q), ("ま", 67, Q), ("れ", 67, H), ("R", 0, Q),
    ]),
    "scale": (96, "音階とあいうえお", [
        ("あ", 60, Q), ("い", 62, Q), ("う", 64, Q), ("え", 65, Q),
        ("お", 67, Q), ("あ", 69, Q), ("い", 71, Q), ("う", 72, H), ("R", 0, Q),
        ("あ", 72, H), ("い", 72, H), ("う", 72, H), ("え", 72, H), ("お", 72, H),
    ]),
}


def write_ust(name: str, tempo: float, notes):
    lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]",
             f"Tempo={tempo:.2f}", "Tracks=1", f"ProjectName={name}", "VoiceDir=", "Mode2=True"]
    for i, (lyric, note, length) in enumerate(notes):
        lines += [f"[#{i:04d}]", f"Length={length}", f"Lyric={lyric}",
                  f"NoteNum={note if note else 60}", "PreUtterance=", "Intensity=100", "Modulation=0"]
    lines.append("[#TRACKEND]")
    path = SONGS / f"{name}.ust"
    path.write_text("\n".join(lines), encoding="shift_jis")
    return path


def main():
    SONGS.mkdir(exist_ok=True)
    for key, (tempo, title, notes) in DEFS.items():
        p = write_ust(key, tempo, notes)
        print(f"[songs] {title}: {len(notes)}音符 -> {p.name}")


if __name__ == "__main__":
    main()
