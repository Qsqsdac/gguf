use super::{DEFAULT_ALIGNMENT, GGufFileType, GGufMetaDataValueType as Ty, GGufMetaValueArray};
use crate::{GGufReadError, GGufReader};

pub trait GGufMetaMap {
    fn get(&self, key: &str) -> Option<(Ty, &[u8])>;
}

#[derive(Debug)]
pub enum GGufMetaError {
    NotExist,
    TypeMismatch(Ty),
    ArrTypeMismatch(Ty),
    OutOfRange,
    Read(GGufReadError),
}

pub trait GGufMetaMapExt: GGufMetaMap {
    fn get_str(&self, key: &str) -> Result<&str, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        match ty {
            Ty::String => GGufReader::new(val).read_str().map_err(GGufMetaError::Read),
            _ => Err(GGufMetaError::TypeMismatch(ty)),
        }
    }

    fn get_usize(&self, key: &str) -> Result<usize, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;

        macro_rules! read {
            ($ty:ty) => {
                GGufReader::new(val)
                    .read::<$ty>()
                    .map_err(GGufMetaError::Read)?
            };
        }
        macro_rules! convert {
            ($val:expr) => {
                $val.try_into().map_err(|_| GGufMetaError::OutOfRange)?
            };
        }

        #[rustfmt::skip]
        let ans = match ty {
            Ty::U8   =>          read!(u8 ).into(),
            Ty::U16  =>          read!(u16).into(),
            Ty::U32  => convert!(read!(u32)      ),
            Ty::U64  => convert!(read!(u64)      ),
            Ty::I8   => convert!(read!(i8 )      ),
            Ty::I16  => convert!(read!(i16)      ),
            Ty::I32  => convert!(read!(i32)      ),
            Ty::I64  => convert!(read!(i64)      ),
            Ty::Bool => if read!(bool) { 1 } else { 0 },
            _        => return Err(GGufMetaError::TypeMismatch(ty)),
        };

        Ok(ans)
    }

    fn get_f32(&self, key: &str) -> Result<f32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::F32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_u32(&self, key: &str) -> Result<u32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::U32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_bool(&self, key: &str) -> Result<bool, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::Bool {
            GGufReader::new(val)
                .read_bool()
                .map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_str_arr(&self, key: &str) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::String {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    fn get_i32_arr(&self, key: &str) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::I32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    fn get_f32_arr(&self, key: &str) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::F32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    #[inline]
    fn general_architecture(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.architecture")
    }

    #[inline]
    fn general_quantization_version(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.quantization_version")
    }

    #[inline]
    fn general_alignment(&self) -> Result<usize, GGufMetaError> {
        match self.get_usize("general.alignment") {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => Ok(DEFAULT_ALIGNMENT),
            Err(e) => Err(e),
        }
    }

    #[inline]
    fn general_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.name")
    }

    #[inline]
    fn general_author(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.author")
    }

    #[inline]
    fn general_version(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.version")
    }

    #[inline]
    fn general_organization(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.organization")
    }

    #[inline]
    fn general_basename(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.basename")
    }

    #[inline]
    fn general_finetune(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.finetune")
    }

    #[inline]
    fn general_description(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.description")
    }

    #[inline]
    fn general_quantized_by(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.quantized_by")
    }

    #[inline]
    fn general_size_label(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.size_label")
    }

    #[inline]
    fn general_license(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license")
    }

    #[inline]
    fn general_license_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.name")
    }

    #[inline]
    fn general_license_link(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.link")
    }

    #[inline]
    fn general_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.url")
    }

    #[inline]
    fn general_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.doi")
    }

    #[inline]
    fn general_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.uuid")
    }

    #[inline]
    fn general_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.repo_url")
    }

    #[inline]
    fn general_tags(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.tags")
    }

    #[inline]
    fn general_languages(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.languages")
    }

    #[inline]
    fn general_datasets(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.datasets")
    }

    #[inline]
    fn general_filetype(&self) -> Result<GGufFileType, GGufMetaError> {
        (self.get_usize("general.filetype")? as u32)
            .try_into()
            .map_err(|_| GGufMetaError::OutOfRange)
    }

    #[inline]
    fn general_source_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.url")
    }

    #[inline]
    fn general_source_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.doi")
    }

    #[inline]
    fn general_source_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.uuid")
    }

    #[inline]
    fn general_source_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.repo_url")
    }

    #[inline]
    fn general_base_model_count(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.base_model.count")
    }

    #[inline]
    fn general_base_model_name(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.name"))
    }

    #[inline]
    fn general_base_model_author(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.author"))
    }

    #[inline]
    fn general_base_model_version(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.version"))
    }

    #[inline]
    fn general_base_model_organization(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.organization"))
    }

    #[inline]
    fn general_base_model_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.url"))
    }

    #[inline]
    fn general_base_model_doi(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.doi"))
    }

    #[inline]
    fn general_base_model_uuid(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.uuid"))
    }

    #[inline]
    fn general_base_model_repo_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.repo_url"))
    }

    #[inline]
    fn llm_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.context_length"))
    }

    #[inline]
    fn llm_embedding_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.embedding_length"))
    }

    #[inline]
    fn llm_block_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.block_count"))
    }

    #[inline]
    fn llm_feed_forward_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.feed_forward_length"))
    }

    #[inline]
    fn llm_use_parallel_residual(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.use_parallel_residual"))
    }

    #[inline]
    fn llm_tensor_data_layout(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.tensor_data_layout"))
    }

    #[inline]
    fn llm_expert_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_count"))
    }

    #[inline]
    fn llm_expert_used_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_used_count"))
    }

    #[inline]
    fn llm_attention_head_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.head_count"))
    }

    #[inline]
    fn llm_attention_head_count_kv(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.head_count_kv")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => self.llm_attention_head_count(),
            Err(e) => Err(e),
        }
    }

    #[inline]
    fn llm_attention_max_alibi_bias(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.max_alibi_bias"))
    }

    #[inline]
    fn llm_attention_clamp_kqv(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.clamp_kqv"))
    }

    #[inline]
    fn llm_attention_layer_norm_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_epsilon"))
    }

    #[inline]
    fn llm_attention_layer_norm_rms_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_rms_epsilon"))
    }

    #[inline]
    fn llm_attention_key_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.key_length")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => {
                let n_embed = self.llm_embedding_length()?;
                let n_head = self.llm_attention_head_count()?;
                Ok(n_embed / n_head)
            }
            Err(e) => Err(e),
        }
    }

    #[inline]
    fn llm_attention_value_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.value_length")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => {
                let n_embed = self.llm_embedding_length()?;
                let n_head = self.llm_attention_head_count()?;
                Ok(n_embed / n_head)
            }
            Err(e) => Err(e),
        }
    }

    #[inline]
    fn llm_rope_dimension_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.dimension_count"))
    }

    #[inline]
    fn llm_rope_freq_base(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.freq_base"))
    }

    #[inline]
    fn llm_rope_scaling_type(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.rope.scaling.type"))
    }

    #[inline]
    fn llm_rope_scaling_factor(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scaling.factor"))
    }

    #[inline]
    fn llm_rope_scaling_original_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.scaling.original_context_length"))
    }

    #[inline]
    fn llm_rope_scaling_finetuned(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.rope.scaling.finetuned"))
    }

    #[inline]
    fn llm_rope_scale_linear(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scale_linear"))
    }

    #[inline]
    fn llm_ssm_conv_kernel(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.conv_kernel"))
    }

    #[inline]
    fn llm_ssm_inner_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.inner_size"))
    }

    #[inline]
    fn llm_ssm_state_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.state_size"))
    }

    #[inline]
    fn llm_ssm_time_step_rank(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.time_step_rank"))
    }

    #[inline]
    fn tokenizer_ggml_model(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.ggml.model")
    }

    #[inline]
    fn tokenizer_ggml_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.tokens")
    }

    #[inline]
    fn tokenizer_ggml_scores(&self) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        self.get_f32_arr("tokenizer.ggml.scores")
    }

    #[inline]
    fn tokenizer_ggml_token_type(&self) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        self.get_i32_arr("tokenizer.ggml.token_type")
    }

    #[inline]
    fn tokenizer_ggml_merges(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.merges")
    }

    #[inline]
    fn tokenizer_ggml_added_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.added_tokens")
    }

    #[inline]
    fn tokenizer_ggml_bos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.bos_token_id")
    }

    #[inline]
    fn tokenizer_ggml_eos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.eos_token_id")
    }

    #[inline]
    fn tokenizer_ggml_unknown_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.unknown_token_id")
    }

    #[inline]
    fn tokenizer_ggml_separator_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.separator_token_id")
    }

    #[inline]
    fn tokenizer_ggml_padding_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.padding_token_id")
    }

    #[inline]
    fn tokenizer_rwkv_world(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.rwkv.world")
    }

    #[inline]
    fn tokenizer_chat_template(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.chat_template")
    }
}

impl<T: GGufMetaMap> GGufMetaMapExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // 实现一个简单的 MetaMap 用于测试
    struct TestMetaMap {
        data: HashMap<String, (Ty, Vec<u8>)>,
    }

    impl TestMetaMap {
        fn new() -> Self {
            let mut map = HashMap::new();

            // 添加各种类型的测试数据

            // 通用字段
            map.insert(
                "general.architecture".to_string(),
                (Ty::String, encode_string("llama")),
            );
            map.insert(
                "general.name".to_string(),
                (Ty::String, encode_string("TestModel")),
            );
            map.insert(
                "general.author".to_string(),
                (Ty::String, encode_string("Test Author")),
            );
            map.insert(
                "general.version".to_string(),
                (Ty::String, encode_string("1.0")),
            );
            map.insert(
                "general.description".to_string(),
                (Ty::String, encode_string("Test Description")),
            );
            map.insert(
                "general.quantization_version".to_string(),
                (Ty::U32, encode_u32(2)),
            );
            map.insert("general.alignment".to_string(), (Ty::U32, encode_u32(32)));
            map.insert(
                "general.size_label".to_string(),
                (Ty::String, encode_string("7B")),
            );
            map.insert(
                "general.finetune".to_string(),
                (Ty::String, encode_string("chat")),
            );
            map.insert(
                "general.tags".to_string(),
                (Ty::Array, encode_string_array(&["tag1", "tag2"])),
            );
            map.insert(
                "general.languages".to_string(),
                (Ty::Array, encode_string_array(&["en", "zh"])),
            );

            // LLM 特定字段
            map.insert(
                "llama.context_length".to_string(),
                (Ty::U32, encode_u32(4096)),
            );
            map.insert(
                "llama.embedding_length".to_string(),
                (Ty::U32, encode_u32(4096)),
            );
            map.insert("llama.block_count".to_string(), (Ty::U32, encode_u32(32)));
            map.insert(
                "llama.feed_forward_length".to_string(),
                (Ty::U32, encode_u32(11008)),
            );
            map.insert(
                "llama.attention.head_count".to_string(),
                (Ty::U32, encode_u32(32)),
            );
            map.insert(
                "llama.rope.dimension_count".to_string(),
                (Ty::U32, encode_u32(128)),
            );
            map.insert(
                "llama.rope.freq_base".to_string(),
                (Ty::F32, encode_f32(10000.0)),
            );
            map.insert(
                "llama.use_parallel_residual".to_string(),
                (Ty::Bool, encode_bool(true)),
            );

            // 分词器相关字段
            map.insert(
                "tokenizer.ggml.model".to_string(),
                (Ty::String, encode_string("llama")),
            );
            map.insert(
                "tokenizer.ggml.bos_token_id".to_string(),
                (Ty::U32, encode_u32(1)),
            );
            map.insert(
                "tokenizer.ggml.eos_token_id".to_string(),
                (Ty::U32, encode_u32(2)),
            );
            map.insert(
                "tokenizer.chat_template".to_string(),
                (
                    Ty::String,
                    encode_string("<|im_start|>user\n{prompt}<|im_end|>"),
                ),
            );

            // attention 相关字段
            map.insert(
                "llama.attention.layer_norm_epsilon".to_string(),
                (Ty::F32, encode_f32(1e-6)),
            );
            map.insert(
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                (Ty::F32, encode_f32(1e-5)),
            );
            map.insert(
                "llama.attention.max_alibi_bias".to_string(),
                (Ty::F32, encode_f32(8.0)),
            );
            map.insert(
                "llama.attention.clamp_kqv".to_string(),
                (Ty::F32, encode_f32(0.0)),
            );

            // rope 相关字段
            map.insert(
                "llama.rope.scaling.type".to_string(),
                (Ty::String, encode_string("linear")),
            );
            map.insert(
                "llama.rope.scaling.factor".to_string(),
                (Ty::F32, encode_f32(1.0)),
            );
            map.insert(
                "llama.rope.scaling.original_context_length".to_string(),
                (Ty::U32, encode_u32(2048)),
            );
            map.insert(
                "llama.rope.scaling.finetuned".to_string(),
                (Ty::Bool, encode_bool(false)),
            );
            map.insert(
                "llama.rope.scale_linear".to_string(),
                (Ty::F32, encode_f32(1.0)),
            );

            // ssm 相关字段
            map.insert(
                "llama.ssm.conv_kernel".to_string(),
                (Ty::U32, encode_u32(4)),
            );
            map.insert(
                "llama.ssm.inner_size".to_string(),
                (Ty::U32, encode_u32(256)),
            );
            map.insert(
                "llama.ssm.state_size".to_string(),
                (Ty::U32, encode_u32(16)),
            );
            map.insert(
                "llama.ssm.time_step_rank".to_string(),
                (Ty::U32, encode_u32(8)),
            );

            Self { data: map }
        }
    }

    impl GGufMetaMap for TestMetaMap {
        fn get(&self, key: &str) -> Option<(Ty, &[u8])> {
            self.data.get(key).map(|(ty, data)| (*ty, data.as_slice()))
        }
    }

    // 辅助函数，用于编码各种数据类型
    fn encode_string(s: &str) -> Vec<u8> {
        let mut result = Vec::new();
        let len = s.len() as u64;
        result.extend_from_slice(&len.to_le_bytes());
        result.extend_from_slice(s.as_bytes());
        result
    }

    fn encode_u32(val: u32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn encode_f32(val: f32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn encode_bool(val: bool) -> Vec<u8> {
        vec![if val { 1u8 } else { 0u8 }]
    }

    fn encode_string_array(strings: &[&str]) -> Vec<u8> {
        let mut result = Vec::new();

        // 写入数组头部
        let element_type = Ty::String as u32;
        result.extend_from_slice(&element_type.to_le_bytes());

        let len = strings.len() as u64;
        result.extend_from_slice(&len.to_le_bytes());

        // 写入每个字符串
        for s in strings {
            let str_len = s.len() as u64;
            result.extend_from_slice(&str_len.to_le_bytes());
            result.extend_from_slice(s.as_bytes());
        }

        result
    }

    #[test]
    fn test_general_fields() {
        let meta_map = TestMetaMap::new();

        assert_eq!(meta_map.general_architecture().unwrap(), "llama");
        assert_eq!(meta_map.general_name().unwrap(), "TestModel");
        assert_eq!(meta_map.general_author().unwrap(), "Test Author");
        assert_eq!(meta_map.general_version().unwrap(), "1.0");
        assert_eq!(meta_map.general_description().unwrap(), "Test Description");
        assert_eq!(meta_map.general_quantization_version().unwrap(), 2);
        assert_eq!(meta_map.general_alignment().unwrap(), 32);
        assert_eq!(meta_map.general_size_label().unwrap(), "7B");
        assert_eq!(meta_map.general_finetune().unwrap(), "chat");
    }

    #[test]
    fn test_llm_fields() {
        let meta_map = TestMetaMap::new();

        // 基本字段测试
        assert_eq!(meta_map.llm_context_length().unwrap(), 4096);
        assert_eq!(meta_map.llm_embedding_length().unwrap(), 4096);
        assert_eq!(meta_map.llm_block_count().unwrap(), 32);
        assert_eq!(meta_map.llm_feed_forward_length().unwrap(), 11008);
        assert_eq!(meta_map.llm_use_parallel_residual().unwrap(), true);

        // attention 相关字段测试
        assert_eq!(meta_map.llm_attention_head_count().unwrap(), 32);
        assert_eq!(meta_map.llm_attention_head_count_kv().unwrap(), 32);
        assert_eq!(meta_map.llm_attention_layer_norm_epsilon().unwrap(), 1e-6);
        assert_eq!(
            meta_map.llm_attention_layer_norm_rms_epsilon().unwrap(),
            1e-5
        );
        assert_eq!(meta_map.llm_attention_max_alibi_bias().unwrap(), 8.0);
        assert_eq!(meta_map.llm_attention_clamp_kqv().unwrap(), 0.0);

        // rope 相关字段测试
        assert_eq!(meta_map.llm_rope_dimension_count().unwrap(), 128);
        assert_eq!(meta_map.llm_rope_freq_base().unwrap(), 10000.0);
        assert_eq!(meta_map.llm_rope_scaling_type().unwrap(), "linear");
        assert_eq!(meta_map.llm_rope_scaling_factor().unwrap(), 1.0);
        assert_eq!(
            meta_map.llm_rope_scaling_original_context_length().unwrap(),
            2048
        );
        assert_eq!(meta_map.llm_rope_scaling_finetuned().unwrap(), false);
        assert_eq!(meta_map.llm_rope_scale_linear().unwrap(), 1.0);

        // ssm 相关字段测试
        assert_eq!(meta_map.llm_ssm_conv_kernel().unwrap(), 4);
        assert_eq!(meta_map.llm_ssm_inner_size().unwrap(), 256);
        assert_eq!(meta_map.llm_ssm_state_size().unwrap(), 16);
        assert_eq!(meta_map.llm_ssm_time_step_rank().unwrap(), 8);
    }

    #[test]
    fn test_tokenizer_fields() {
        let meta_map = TestMetaMap::new();

        assert_eq!(meta_map.tokenizer_ggml_model().unwrap(), "llama");
        assert_eq!(meta_map.tokenizer_ggml_bos_token_id().unwrap(), 1);
        assert_eq!(meta_map.tokenizer_ggml_eos_token_id().unwrap(), 2);
        assert_eq!(
            meta_map.tokenizer_chat_template().unwrap(),
            "<|im_start|>user\n{prompt}<|im_end|>"
        );
    }

    #[test]
    fn test_array_fields() {
        let meta_map = TestMetaMap::new();

        let tags: Vec<_> = meta_map
            .general_tags()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(tags, vec!["tag1", "tag2"]);

        let languages: Vec<_> = meta_map
            .general_languages()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(languages, vec!["en", "zh"]);
    }

    #[test]
    fn test_missing_fields() {
        let meta_map = TestMetaMap::new();

        // 对于未定义的字段，我们期望得到 NotExist 错误
        assert!(matches!(
            meta_map.general_license().unwrap_err(),
            GGufMetaError::NotExist
        ));
        assert!(matches!(
            meta_map.general_url().unwrap_err(),
            GGufMetaError::NotExist
        ));
        assert!(meta_map.llm_attention_head_count_kv().is_ok()); // 这个会回退到 head_count
    }

    #[test]
    fn test_calculated_fields() {
        let meta_map = TestMetaMap::new();

        // 测试那些有默认计算逻辑的字段
        assert_eq!(meta_map.llm_attention_key_length().unwrap(), 128); // 4096 / 32
        assert_eq!(meta_map.llm_attention_value_length().unwrap(), 128); // 4096 / 32

        // 确保回退机制正常工作
        assert_eq!(meta_map.llm_attention_head_count_kv().unwrap(), 32); // 回退到 head_count
    }

    #[test]
    fn test_error_handling() {
        // 创建一个特殊的测试 map，添加一些错误的类型
        let mut map = HashMap::new();
        map.insert("wrong_type_field".to_string(), (Ty::String, encode_u32(42)));
        map.insert(
            "array_wrong_type".to_string(),
            (Ty::Array, {
                let mut data = Vec::new();
                // 写入类型为 U16 的数组头部
                data.extend_from_slice(&(Ty::U16 as u32).to_le_bytes());
                data.extend_from_slice(&(2u64).to_le_bytes());
                data
            }),
        );

        let meta_map = TestMetaMap { data: map };

        // 对于错误的数据编码
        let result = meta_map.get_str("wrong_type_field");
        assert!(result.is_err());
        assert!(matches!(result, Err(GGufMetaError::Read(_))));

        // 对于数组类型不匹配的情况
        let result = meta_map.get_str_arr("array_wrong_type");
        assert!(result.is_err());
        if let Err(err) = result {
            match err {
                GGufMetaError::ArrTypeMismatch(ty) => {
                    assert_eq!(ty, Ty::U16);
                }
                _ => panic!("Expected ArrTypeMismatch, got {:?}", err),
            }
        }

        // 测试字段不存在的情况
        let result = meta_map.get_str("non_existent_field");
        assert!(result.is_err());
        assert!(matches!(result, Err(GGufMetaError::NotExist)));
    }
}
