use alloc::vec::Vec;

pub trait DiffusionMatrix {
    fn divisor(&self) -> usize;
    fn targets(&self) -> &[(isize, usize, usize)];
}

impl DiffusionMatrix for alloc::boxed::Box<dyn DiffusionMatrix> {
    fn divisor(&self) -> usize {
        self.as_ref().divisor()
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        self.as_ref().targets()
    }
}

pub struct NoDiffuse;
impl DiffusionMatrix for NoDiffuse {
    fn divisor(&self) -> usize {
        1
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        &[]
    }
}

pub struct FloydSteinberg;
impl DiffusionMatrix for FloydSteinberg {
    fn divisor(&self) -> usize {
        16
    }

    fn targets(&self) -> &[(isize, usize, usize)] {
        &[(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]
    }
}

pub struct JarvisJudiceAndNinke;
impl DiffusionMatrix for JarvisJudiceAndNinke {
    fn divisor(&self) -> usize {
        48
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        &[
            // First row
            (1, 0, 7),
            (2, 0, 5),
            // Second row
            (-2, 1, 3),
            (-1, 1, 5),
            (0, 1, 7),
            (1, 1, 5),
            (2, 1, 3),
            // Third row
            (-2, 2, 1),
            (-1, 2, 3),
            (0, 2, 5),
            (1, 2, 3),
            (2, 2, 1),
        ]
    }
}

pub struct Atkinson;
impl DiffusionMatrix for Atkinson {
    fn divisor(&self) -> usize {
        8
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        &[
            // First row
            (1, 0, 1),
            (2, 0, 1),
            // Second row
            (-1, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
            // Third row
            (0, 2, 1),
        ]
    }
}

pub struct Sierra;
impl DiffusionMatrix for Sierra {
    fn divisor(&self) -> usize {
        32
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        &[
            // First row
            (1, 0, 5),
            (2, 0, 3),
            // Second row
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 5),
            (1, 1, 4),
            (2, 1, 2),
            // Third row
            (-1, 2, 2),
            (0, 2, 3),
            (1, 2, 2),
        ]
    }
}

pub struct DynamicDiffusionMatrix(Vec<(isize, usize, usize)>, usize);
impl DiffusionMatrix for DynamicDiffusionMatrix {
    fn divisor(&self) -> usize {
        self.1
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        self.0.as_slice()
    }
}
impl DynamicDiffusionMatrix {
    fn new<T: DiffusionMatrix>(t: T) -> Self {
        Self(Vec::from(t.targets()), t.divisor())
    }
}
