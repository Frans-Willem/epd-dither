pub(crate) fn opt_array_transpose<T, const N: usize>(arr: [Option<T>; N]) -> Option<[T; N]> {
    if arr.iter().all(Option::is_some) {
        Some(arr.map(Option::unwrap))
    } else {
        None
    }
}

/// `Ord::max` analogue for `T: PartialOrd`. Returns `a` on a tie or on
/// incomparable values (e.g. NaN), matching the convention `if a > b { a } else { b }`.
pub(crate) fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// `Ord::clamp` analogue for `T: PartialOrd`. Incomparable inputs (e.g. NaN)
/// pass through unchanged.
pub(crate) fn partial_clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}
