use crate::*;

pub struct Distribution1D {
    pub pmf: Vec<Float>,
    cdf: Vec<Float>,
}

// the first i s.t. v <= f[i]
pub fn upper_bound<T: PartialOrd>(f: &[T], v: &T) -> usize {
    let mut lo = 0;
    let mut hi = f.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if f[mid] < *v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use statrs::assert_almost_eq;

    use crate::distribution::upper_bound;
    #[test]
    fn test_distr() {
        use crate::distribution::Distribution1D;
        use crate::Float;
        let dist = Distribution1D::new(&[0.5, 0.5, 1.0, 1.0]).unwrap();
        let f_int: Float = 3.0;
        assert_almost_eq!(dist.pdf_discrete(0) as f64, (0.5 / f_int) as f64, 0.001f64);
        assert_almost_eq!(dist.pdf_discrete(3) as f64, (1.0 / f_int) as f64, 0.001f64);
        {
            let (i, pdf) = dist.sample_discrete(0.7);
            assert_eq!(i, 3);
            assert_almost_eq!(pdf as f64, (1.0 / f_int) as f64, 0.001f64);
        }
        {
            let (i, pdf) = dist.sample_discrete(0.1);
            assert_eq!(i, 0);
            assert_almost_eq!(pdf as f64, (0.5 / f_int) as f64, 0.001f64);
        }
    }
    #[test]
    fn upper_bounds() {
        assert_eq!(upper_bound(&[0, 1, 2, 3, 4, 5, 6, 7], &7), 7);
        assert_eq!(
            upper_bound(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &0.12),
            1
        );
        assert_eq!(
            upper_bound(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &0.22),
            2
        );
        assert_eq!(
            upper_bound(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &0.32),
            3
        );
        assert_eq!(
            upper_bound(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &0.72),
            7
        );
        assert_eq!(
            upper_bound(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &0.9),
            7
        );
    }
}

impl Distribution1D {
    pub fn new(f: &[Float]) -> Option<Self> {
        let mut f: Vec<_> = Vec::from(f);
        let int_f: Float = f.iter().map(|x| *x).sum::<Float>(); // / f.len() as Float;
        if int_f == 0.0 {
            return None;
        }
        f.iter_mut().for_each(|x| *x /= int_f);
        let mut cdf = vec![];
        let mut sum = 0.0;
        cdf.push(0.0);
        for x in &f {
            sum += *x;
            cdf.push(sum);
        }
        Some(Self { pmf: f, cdf })
    }
    pub fn pdf_discrete(&self, idx: usize) -> Float {
        self.pmf[idx]
    }

    pub fn sample_discrete(&self, u: Float) -> (usize, Float) {
        let i = upper_bound(&self.cdf[..], &u);
        let i = if i == 0 { 0 } else { i - 1 };
        (i, self.pmf[i])
    }
    pub fn sample_continuous(&self, u: Float) -> (Float, Float) {
        let i = upper_bound(&self.cdf[..], &u) - 1;
        let i = if i == 0 { 0 } else { i - 1 };
        let cnt = self.pmf.len() as Float;
        let du = u - self.cdf[i];
        let du = if self.cdf[i + 1] > self.cdf[i] {
            du / (self.cdf[i + 1] - self.cdf[i])
        } else {
            du
        };
        ((i as Float + du) / cnt, self.pmf[i] * cnt)
    }
}