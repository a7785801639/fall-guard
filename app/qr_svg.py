from __future__ import annotations

"""
Автономный zero-dependency генератор валидного QR-кода (ISO/IEC 18004, Byte mode,
Reed-Solomon GF(256), версии 1..6 с автопереключением ECC M -> L, выбор оптимальной
маски по 4 правилам штрафов и точные BCH(15,5) биты формата) в компактный векторный SVG.
"""

# Таблицы GF(256) по примитивному полиному QR-кода x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256


def _init_gf256() -> None:
    val = 1
    for i in range(255):
        _GF_EXP[i] = val
        _GF_LOG[val] = i
        val <<= 1
        if val & 0x100:
            val ^= 0x11D
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_init_gf256()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator_poly(degree: int) -> list[int]:
    poly = [1]
    for i in range(degree):
        # Умножаем poly на (x - alpha^i); в GF(2^8) сложение и вычитание эквивалентны XOR
        nxt = [0] * (len(poly) + 1)
        root = _GF_EXP[i]
        for j, coeff in enumerate(poly):
            nxt[j] ^= coeff
            nxt[j + 1] ^= _gf_mul(coeff, root)
        poly = nxt
    return poly


def _rs_encode_block(data: list[int], ec_len: int) -> list[int]:
    gen = _rs_generator_poly(ec_len)
    reg = [0] * ec_len
    for byte in data:
        factor = byte ^ reg[0]
        reg = reg[1:] + [0]
        if factor != 0:
            for j in range(ec_len):
                reg[j] ^= _gf_mul(gen[j + 1], factor)
    return reg


# Спецификация версий 1..6 ISO/IEC 18004:
# (version, ec_format_bits, total_data_codewords, ec_per_block, num_blocks, remainder_bits)
# ec_format_bits: M = 0b00 (0), L = 0b01 (1)
_VERSION_SPECS_M: tuple[tuple[int, int, int, int, int, int], ...] = (
    (1, 0b00, 16, 10, 1, 0),
    (2, 0b00, 28, 16, 1, 7),
    (3, 0b00, 44, 26, 1, 7),
    (4, 0b00, 64, 18, 2, 7),
    (5, 0b00, 86, 24, 2, 7),
    (6, 0b00, 108, 16, 4, 7),
)

_VERSION_SPECS_L: tuple[tuple[int, int, int, int, int, int], ...] = (
    (1, 0b01, 19, 7, 1, 0),
    (2, 0b01, 34, 10, 1, 7),
    (3, 0b01, 55, 15, 1, 7),
    (4, 0b01, 80, 20, 1, 7),
    (5, 0b01, 108, 26, 1, 7),
    (6, 0b01, 136, 18, 2, 7),
)

_ALIGNMENT_COORDS: dict[int, tuple[int, ...]] = {
    1: (),
    2: (6, 18),
    3: (6, 22),
    4: (6, 26),
    5: (6, 30),
    6: (6, 34),
}


def _select_version_spec(payload_len: int) -> tuple[int, int, int, int, int, int]:
    # В Byte mode для версий 1..9 заголовок занимает 4 (mode) + 8 (count) = 12 бит -> вместимость data_cw - 2 байт
    for spec in _VERSION_SPECS_M:
        if payload_len <= spec[2] - 2:
            return spec
    for spec in _VERSION_SPECS_L:
        if payload_len <= spec[2] - 2:
            return spec
    raise ValueError(f"Текст слишком длинный для QR-кода версий 1..6 ({payload_len} байт, макс. 134)")


def _build_codewords(
    payload: bytes,
    total_data_cw: int,
    ec_per_block: int,
    num_blocks: int,
) -> list[int]:
    bits: list[int] = []

    def append_bits(val: int, length: int) -> None:
        for shift in range(length - 1, -1, -1):
            bits.append((val >> shift) & 1)

    # Byte mode indicator = 0100, character count = 8 bits (версии 1..9)
    append_bits(0b0100, 4)
    append_bits(len(payload), 8)
    for b in payload:
        append_bits(b, 8)

    capacity_bits = total_data_cw * 8
    term_len = min(4, capacity_bits - len(bits))
    append_bits(0, term_len)

    while len(bits) % 8 != 0:
        bits.append(0)

    data_cw: list[int] = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | bit
        data_cw.append(byte_val)

    pad_bytes = (0xEC, 0x11)
    pad_idx = 0
    while len(data_cw) < total_data_cw:
        data_cw.append(pad_bytes[pad_idx % 2])
        pad_idx += 1

    block_data_len = total_data_cw // num_blocks
    data_blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    offset = 0
    for _ in range(num_blocks):
        chunk = data_cw[offset : offset + block_data_len]
        offset += block_data_len
        data_blocks.append(chunk)
        ec_blocks.append(_rs_encode_block(chunk, ec_per_block))

    # Чередование (interleaving) кодовых слов данных и коррекции ошибок
    interleaved: list[int] = []
    for col in range(block_data_len):
        for blk in data_blocks:
            interleaved.append(blk[col])
    for col in range(ec_per_block):
        for eblk in ec_blocks:
            interleaved.append(eblk[col])
    return interleaved


def _format_info_bits(ec_bits: int, mask_id: int) -> int:
    data5 = ((ec_bits & 0b11) << 3) | (mask_id & 0b111)
    rem = data5 << 10
    for shift in range(14, 9, -1):
        if (rem >> shift) & 1:
            rem ^= 0x537 << (shift - 10)
    return ((data5 << 10) | rem) ^ 0x5412


def _place_function_patterns(version: int) -> tuple[list[list[int]], list[list[bool]]]:
    size = 17 + 4 * version
    modules = [[0] * size for _ in range(size)]
    is_func = [[False] * size for _ in range(size)]

    def set_func(r: int, c: int, dark: bool) -> None:
        if 0 <= r < size and 0 <= c < size:
            modules[r][c] = 1 if dark else 0
            is_func[r][c] = True

    def draw_finder(top: int, left: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                if 0 <= dr <= 6 and 0 <= dc <= 6:
                    on_border = dr in (0, 6) or dc in (0, 6)
                    in_center = 2 <= dr <= 4 and 2 <= dc <= 4
                    set_func(r, c, on_border or in_center)
                else:
                    set_func(r, c, False)

    draw_finder(0, 0)
    draw_finder(0, size - 7)
    draw_finder(size - 7, 0)

    # Полосы синхронизации (Timing patterns)
    for i in range(8, size - 8):
        dark = (i % 2) == 0
        set_func(6, i, dark)
        set_func(i, 6, dark)

    # Выравнивающие паттерны (Alignment patterns)
    coords = _ALIGNMENT_COORDS[version]
    last_idx = len(coords) - 1
    for ir, cr in enumerate(coords):
        for ic, cc in enumerate(coords):
            if (ir == 0 and ic == 0) or (ir == 0 and ic == last_idx) or (ir == last_idx and ic == 0):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    dist = max(abs(dr), abs(dc))
                    set_func(cr + dr, cc + dc, dist != 1)

    # Резервируем ячейки под Format Information + обязательный Dark module
    for i in range(9):
        if i != 6:
            set_func(8, i, False)
            set_func(i, 8, False)
    for i in range(8):
        set_func(8, size - 1 - i, False)
        set_func(size - 1 - i, 8, False)
    set_func(size - 8, 8, True)  # Dark module (4*V + 9, 8)

    return modules, is_func


def _write_format_bits(modules: list[list[int]], ec_bits: int, mask_id: int) -> None:
    size = len(modules)
    bits = _format_info_bits(ec_bits, mask_id)

    # Первая копия вокруг верхнего левого поискового паттерна (бит 0 = LSB .. бит 14 = MSB)
    for i in range(6):
        modules[i][8] = (bits >> i) & 1
    modules[7][8] = (bits >> 6) & 1
    modules[8][8] = (bits >> 7) & 1
    modules[8][7] = (bits >> 8) & 1
    for i in range(9, 15):
        modules[8][14 - i] = (bits >> i) & 1

    # Вторая копия (верхний правый + нижний левый углы)
    for i in range(8):
        modules[8][size - 1 - i] = (bits >> i) & 1
    for i in range(8, 15):
        modules[size - 15 + i][8] = (bits >> i) & 1


def _mask_bit(mask_id: int, r: int, c: int) -> bool:
    if mask_id == 0:
        return (r + c) % 2 == 0
    if mask_id == 1:
        return r % 2 == 0
    if mask_id == 2:
        return c % 3 == 0
    if mask_id == 3:
        return (r + c) % 3 == 0
    if mask_id == 4:
        return (r // 2 + c // 3) % 2 == 0
    if mask_id == 5:
        return (r * c) % 2 + (r * c) % 3 == 0
    if mask_id == 6:
        return ((r * c) % 2 + (r * c) % 3) % 2 == 0
    return ((r + c) % 2 + (r * c) % 3) % 2 == 0


def _penalty_score(mat: list[list[int]]) -> int:
    size = len(mat)
    score = 0

    # N1: 5+ одинаковых подряд по строкам и столбцам
    for r in range(size):
        run_val = mat[r][0]
        run_len = 1
        for c in range(1, size):
            if mat[r][c] == run_val:
                run_len += 1
            else:
                if run_len >= 5:
                    score += 3 + (run_len - 5)
                run_val = mat[r][c]
                run_len = 1
        if run_len >= 5:
            score += 3 + (run_len - 5)

    for c in range(size):
        run_val = mat[0][c]
        run_len = 1
        for r in range(1, size):
            if mat[r][c] == run_val:
                run_len += 1
            else:
                if run_len >= 5:
                    score += 3 + (run_len - 5)
                run_val = mat[r][c]
                run_len = 1
        if run_len >= 5:
            score += 3 + (run_len - 5)

    # N2: одноцветные блоки 2x2
    for r in range(size - 1):
        row0 = mat[r]
        row1 = mat[r + 1]
        for c in range(size - 1):
            v = row0[c]
            if row0[c + 1] == v and row1[c] == v and row1[c + 1] == v:
                score += 3

    # N3: паттерны 1:1:3:1:1 в соседстве с 4 светлыми модулями
    pat1 = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    pat2 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    for r in range(size):
        line = mat[r]
        for i in range(size - 10):
            seg = line[i : i + 11]
            if seg == pat1 or seg == pat2:
                score += 40
    for c in range(size):
        col_line = [mat[r][c] for r in range(size)]
        for i in range(size - 10):
            seg = col_line[i : i + 11]
            if seg == pat1 or seg == pat2:
                score += 40

    # N4: баланс тёмных и светлых модулей
    dark = sum(sum(row) for row in mat)
    total = size * size
    k = abs(dark * 20 - total * 10) // total
    score += k * 10
    return score


def encode_qr_matrix(text: str) -> list[list[bool]]:
    """Кодирует произвольную UTF-8 строку в булеву матрицу QR-кода (True = тёмный модуль)."""
    if not text:
        raise ValueError("Текст для QR-кода не должен быть пустым")
    payload = text.encode("utf-8")
    version, ec_bits, total_data_cw, ec_per_block, num_blocks, rem_bits = _select_version_spec(
        len(payload)
    )
    codewords = _build_codewords(payload, total_data_cw, ec_per_block, num_blocks)

    bit_stream: list[int] = []
    for cw in codewords:
        for shift in range(7, -1, -1):
            bit_stream.append((cw >> shift) & 1)
    bit_stream.extend([0] * rem_bits)

    base_modules, is_func = _place_function_patterns(version)
    size = len(base_modules)

    # Зигзагообразная укладка бит данных справа налево двухколоночными полосами
    bit_idx = 0
    upward = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for dc in (0, 1):
                c = col - dc
                if not is_func[r][c]:
                    base_modules[r][c] = bit_stream[bit_idx] if bit_idx < len(bit_stream) else 0
                    bit_idx += 1
        upward = not upward
        col -= 2

    best_mat: list[list[int]] | None = None
    best_penalty = 1 << 30
    for mask_id in range(8):
        cand = [row[:] for row in base_modules]
        for r in range(size):
            for c in range(size):
                if not is_func[r][c] and _mask_bit(mask_id, r, c):
                    cand[r][c] ^= 1
        _write_format_bits(cand, ec_bits, mask_id)
        pen = _penalty_score(cand)
        if pen < best_penalty:
            best_penalty = pen
            best_mat = cand

    assert best_mat is not None
    return [[bool(cell) for cell in row] for row in best_mat]


def render_qr_svg(text: str, border: int = 2) -> str:
    """
    Генерирует чистый компактный векторный QR-код <svg> без внешних зависимостей.
    Горизонтально смежные тёмные модули склеиваются в единые subpaths для минимального веса.
    """
    matrix = encode_qr_matrix(text)
    size = len(matrix)
    pad = max(0, int(border))
    total = size + pad * 2

    cmds: list[str] = []
    for r, row in enumerate(matrix):
        y = r + pad
        c = 0
        while c < size:
            if not row[c]:
                c += 1
                continue
            run = 1
            while (c + run) < size and row[c + run]:
                run += 1
            x = c + pad
            cmds.append(f"M{x} {y}h{run}v1h-{run}z")
            c += run

    d_attr = "".join(cmds)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
        f'shape-rendering="crispEdges"><path fill="currentColor" d="{d_attr}"/></svg>'
    )
