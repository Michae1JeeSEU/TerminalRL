"""Prepare an inactive candidate while the server is unavailable."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
original = (root / 'terminal_marl/core/polynomial_trajectory.py').read_bytes()
assert hashlib.sha256(original).hexdigest() == '832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f'
baseline = root / '.tmp/p3_polynomial_baseline_20261005.py'
if baseline.exists():
    assert baseline.read_bytes() == original
else:
    baseline.write_bytes(original)
text = original.decode('utf-8')

def replace(before, after):
    global text
    assert text.count(before) == 1, before[:100]
    text = text.replace(before, after)

replace('values = np.asarray(list(coefficients), dtype=float).reshape(-1)', '''# Internal callers already provide arrays; avoid boxing every coefficient.
    values = np.asarray(
        coefficients if isinstance(coefficients, np.ndarray) else list(coefficients),
        dtype=float,
    ).reshape(-1)''')
replace('''    trimmed = np.array(values[: last + 1], dtype=float, copy=True)
    if not np.any(np.abs(trimmed) > tolerance):
        return np.zeros(1, dtype=float)
    return trimmed''', '''    # The last surviving finite coefficient already proves this is nonzero.
    # Keep the original all-zero/nonfinite rule when that proof does not hold.
    if not abs(values[last]) > tolerance:
        if not np.any(np.abs(values[: last + 1]) > tolerance):
            return np.zeros(1, dtype=float)
    return np.array(values[: last + 1], dtype=float, copy=True)''')
text = text.replace('float(np.clip(real_root, 0.0, 1.0))', 'min(max(real_root, 0.0), 1.0)')
replace('values = sorted(float(np.clip(point, 0.0, 1.0)) for point in points)', 'values = sorted(min(max(float(point), 0.0), 1.0) for point in points)')
replace('''    result = np.zeros(1, dtype=float)
    for axis in axes:
        product = np.polynomial.polynomial.polymul(left[:, axis], right[:, axis])
        if result.size < product.size:
            result = np.pad(result, (0, product.size - result.size))
        result[: product.size] += product
    return _trim_polynomial(result)''', '''    # These internal operands are finite float derivative arrays. Match
    # polymul's exact-zero trimming and convolution order, without its generic
    # dtype/series conversion or repeated padding. Do not round coefficients.
    result = np.zeros(left.shape[0] + right.shape[0] - 1, dtype=float)
    for axis in axes:
        lhs, rhs = left[:, axis], right[:, axis]
        left_last, right_last = lhs.size - 1, rhs.size - 1
        while left_last > 0 and lhs[left_last] == 0.0:
            left_last -= 1
        while right_last > 0 and rhs[right_last] == 0.0:
            right_last -= 1
        product = np.convolve(lhs[: left_last + 1], rhs[: right_last + 1])
        result[: product.size] += product
    return _trim_polynomial(result)''')
replace('''    velocity_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 1) for segment in uncached], axis=0
    )
    acceleration_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 2) for segment in uncached], axis=0
    )
    jerk_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 3) for segment in uncached], axis=0
    )''', '''    # Preserve the scalar sequence of multiplications, across all segments.
    velocity_u = coefficients[:, 1:, :] * np.arange(1, 6, dtype=float)[None, :, None]
    acceleration_u = velocity_u[:, 1:, :] * np.arange(1, 5, dtype=float)[None, :, None]
    jerk_u = acceleration_u[:, 1:, :] * np.arange(1, 4, dtype=float)[None, :, None]''')
(root / '.tmp/p3_polynomial_candidate_20261005.py').write_text(text, encoding='utf-8', newline='\n')
print('baseline', hashlib.sha256(original).hexdigest())
print('candidate', hashlib.sha256(text.encode()).hexdigest())
