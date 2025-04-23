#![doc = include_str!("../README.md")]
#![deny(warnings)]

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait DataBlock: Sized + 'static {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout;
    const COUNT: usize;
    const ZEROS: Self;
}

macro_rules! impl_data_block {
    ($ty:ty = $id:expr; $zero:expr ) => {
        impl DataBlock for $ty {
            #[cfg(feature = "types")]
            const ID: digit_layout::DigitLayout = $id;
            const COUNT: usize = Self::ID.group_size();
            const ZEROS: Self = $zero;
        }
    };
}

use digit_layout::types as ty;
impl_data_block!(u8  = ty::U8 ; 0 );
impl_data_block!(i8  = ty::I8 ; 0 );
impl_data_block!(u16 = ty::U16; 0 );
impl_data_block!(i16 = ty::I16; 0 );
impl_data_block!(u32 = ty::U32; 0 );
impl_data_block!(i32 = ty::I32; 0 );
impl_data_block!(f32 = ty::F32; 0.);
impl_data_block!(u64 = ty::U64; 0 );
impl_data_block!(i64 = ty::I64; 0 );
impl_data_block!(f64 = ty::F64; 0.);

pub trait Quantize<T, const N: usize>: DataBlock {
    fn quantize(data: &[T; N]) -> Self;
    fn dequantize(&self) -> [T; N];
}

impl<Blk, const N: usize> Quantize<f16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[f16; N]) -> Self {
        Self::quantize(&data.map(f16::to_f32))
    }
    #[inline]
    fn dequantize(&self) -> [f16; N] {
        self.dequantize().map(f16::from_f32)
    }
}

impl<Blk, const N: usize> Quantize<bf16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[bf16; N]) -> Self {
        Self::quantize(&data.map(bf16::to_f32))
    }
    #[inline]
    fn dequantize(&self) -> [bf16; N] {
        self.dequantize().map(bf16::from_f32)
    }
}

pub trait QuantExt<T, const N: usize>: Sized {
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError>;
    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeError {
    Indivisible,
    LengthMismatch,
}

impl<Blk, T, const N: usize> QuantExt<T, N> for Blk
where
    Blk: Quantize<T, N> + Send + Sync,
    T: Send + Sync,
{
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError> {
        if src.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if dst.len() != src.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let src = unsafe { from_raw_parts(src.as_ptr().cast::<[T; N]>(), dst.len()) };
        dst.into_par_iter()
            .zip(src)
            .for_each(|(dst, src)| *dst = Blk::quantize(src));
        Ok(())
    }

    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if src.len() != dst.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let dst = unsafe { from_raw_parts_mut(dst.as_mut_ptr().cast::<[T; N]>(), src.len()) };
        src.into_par_iter()
            .zip(dst)
            .for_each(|(src, dst)| *dst = Blk::dequantize(src));
        Ok(())
    }
}

mod structs;
pub use structs::*;

#[cfg(feature = "types")]
pub extern crate digit_layout;

#[cfg(feature = "types")]
pub mod types;

#[cfg(test)]
#[allow(dead_code)]
// 测试工具，仅在测试时使用
pub(crate) mod test_utils {
    use crate::Quantize;
    use std::fmt;

    pub fn test<const N: usize, T: Quantize<f32, N>>(abs: f32, rel: f32) {
        use rand::Rng;
        use std::iter::zip;

        let mut data = [0.0f32; N];
        rand::rng().fill(&mut data[..]);

        let quant = T::quantize(&data);
        let dequant = T::dequantize(&quant);

        let mut ec = ErrorCollector::new(abs, rel);
        for (a, b) in zip(data, dequant) {
            ec.push(Diff::new(a, b))
        }
        println!("{ec}");

        for &i in ec.outliers() {
            println!("{} vs {}", data[i], dequant[i]);
        }

        assert!(ec.outliers().is_empty());
    }

    struct Diff {
        pub abs: f32,
        pub rel: f32,
    }

    impl Diff {
        fn new(a: f32, b: f32) -> Self {
            let abs = (a - b).abs();
            let rel = abs / (a.abs() + b.abs() + f32::EPSILON);
            Self { abs, rel }
        }
    }

    struct ErrorCollector {
        threshold: Diff,
        max_diff: Diff,
        outliers: Vec<usize>,
        count: usize,
    }

    impl ErrorCollector {
        fn new(abs: f32, rel: f32) -> Self {
            Self {
                threshold: Diff { abs, rel },
                max_diff: Diff { abs: 0.0, rel: 0.0 },
                outliers: vec![],
                count: 0,
            }
        }

        fn push(&mut self, diff: Diff) {
            self.max_diff.abs = f32::max(self.max_diff.abs, diff.abs);
            self.max_diff.rel = f32::max(self.max_diff.rel, diff.rel);

            if diff.abs > self.threshold.abs && diff.rel > self.threshold.rel {
                self.outliers.push(self.count);
            }

            self.count += 1;
        }

        fn outliers(&self) -> &[usize] {
            &self.outliers
        }
    }

    impl fmt::Display for ErrorCollector {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "abs: {:.3e}, rel: {:.3e}, outliers: {}/{}",
                self.max_diff.abs,
                self.max_diff.rel,
                self.outliers.len(),
                self.count,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantExt, Quantize};
    use half::f16;
    use rand::Rng;

    type Blk = Q8_1; // 切换测试类型
    const N: usize = 32;

    #[test]
    fn test_block_accuracy() {
        crate::test_utils::test::<N, Blk>(4e-3, 0.0);
    }

    #[test]
    fn test_quant_dequant_slice_ok() {  
        let mut rng = rand::rng();
        let input: Vec<f32> = (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut quantized: Vec<Blk> = (0..(input.len() / N))
            .map(|_| Blk::quantize(&[0.0; N]))
            .collect();
        let mut output = vec![0f32; input.len()];

        Blk::quantize_slice(&mut quantized, &input).unwrap();
        Blk::dequantize_slice(&mut output, &quantized).unwrap();

        for (a, b) in input.iter().zip(&output) {
            assert!((a - b).abs() < 1e-1, "{a} vs {b}");
        }
    }

    #[test]
    fn test_quant_slice_indivisible() {   // 测试量化切片不可整除
        let mut rng = rand::rng();
        let input: Vec<f32> = (0..(N - 1)).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut quantized: Vec<Blk> = (0..1).map(|_| Blk::quantize(&[0.0; N])).collect();
        let err = Blk::quantize_slice(&mut quantized, &input).unwrap_err();
        assert_eq!(err, crate::QuantizeError::Indivisible);
    }

    #[test]
    fn test_quant_slice_length_mismatch() {  // 测试量化切片长度不匹配
        let mut rng = rand::rng();
        let input: Vec<f32> = (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut quantized: Vec<Blk> = (0..3).map(|_| Blk::quantize(&[0.0; N])).collect(); // 应为 2
        let err = Blk::quantize_slice(&mut quantized, &input).unwrap_err();
        assert_eq!(err, crate::QuantizeError::LengthMismatch);
    }

    #[test]
    fn test_dequant_slice_indivisible() {  // 测试反量化切片不可整除
        let mut rng = rand::rng();
        let mut output: Vec<f32> = (0..(N - 1)).map(|_| rng.random_range(-1.0..1.0)).collect();
        let src: Vec<Blk> = (0..1).map(|_| Blk::quantize(&[0.0; N])).collect();
        let err = Blk::dequantize_slice(&mut output, &src).unwrap_err();
        assert_eq!(err, crate::QuantizeError::Indivisible);
    }

    #[test]
    fn test_dequant_slice_length_mismatch() {  // 测试反量化切片长度不匹配
        let mut rng = rand::rng();
        let mut output: Vec<f32> = (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
        let src: Vec<Blk> = (0..3).map(|_| Blk::quantize(&[0.0; N])).collect(); // 应为 2
        let err = Blk::dequantize_slice(&mut output, &src).unwrap_err();
        assert_eq!(err, crate::QuantizeError::LengthMismatch);
    }

    #[test]
    fn test_block_zeros_and_count() {   // 测试全零量化块
        assert_eq!(Blk::COUNT, N);
        let dequant: [f32; N] = Blk::quantize(&[0.0; N]).dequantize();
        assert!(dequant.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_quantize_f16() {
        let mut rng = rand::rng();
        let input: [f16; N] = core::array::from_fn(|_| f16::from_f32(rng.random_range(-1.0..1.0)));
        let blk = <Blk as Quantize<f16, N>>::quantize(&input);
        let dequant = <Blk as Quantize<f16, N>>::dequantize(&blk);
        for (a, b) in input.iter().zip(&dequant) {
            let diff = (a.to_f32() - b.to_f32()).abs();
            assert!(diff < 1e-1, "diff = {diff}");
        }
    }
    
    #[test]
    fn test_quantize_bf16() {
        let mut rng = rand::rng();
        let input: [bf16; N] = core::array::from_fn(|_| bf16::from_f32(rng.random_range(-1.0..1.0)));
        let blk = <Blk as Quantize<bf16, N>>::quantize(&input);
        let dequant = <Blk as Quantize<bf16, N>>::dequantize(&blk);
        for (a, b) in input.iter().zip(&dequant) {
            let diff = (a.to_f32() - b.to_f32()).abs();
            assert!(diff < 1e-1, "diff = {diff}");
        }
    }
}