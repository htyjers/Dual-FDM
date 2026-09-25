# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)





        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, denoise_step=0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.denoise_step = denoise_step
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.style_matrix = None
        self.style_matrix1 = None
        self.style_matrix2 = None
        self.style_matrix3 = None
        self.scale_entropy = 0
        self.layer = 0
        self.wk = []



    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        self.denoise_step += 1
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            print("attn.group_norm")
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query1 = attn.to_q(hidden_states)
        query2 = hidden_states




        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos1 = encoder_hidden_states.shape[1] - self.num_tokens * 4
            end_pos2 = encoder_hidden_states.shape[1] - self.num_tokens * 3
            end_pos3 = encoder_hidden_states.shape[1] - self.num_tokens * 2
            end_pos4 = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, encoder_content_hidden_states, encoder_content_hidden_states1, ip_hidden_states, ip_hidden_states_style, nums , numc = (
                encoder_hidden_states[:, :77, :],
                encoder_hidden_states[:, 77:154, :],
                encoder_hidden_states[:, 154:end_pos1, :],
                encoder_hidden_states[:, end_pos1:end_pos2, :],
                encoder_hidden_states[:, end_pos2:end_pos3, :],
                encoder_hidden_states[:, end_pos3:end_pos4, :],
                encoder_hidden_states[:, end_pos4:, :],
            )

            if attn.norm_cross:
                print("attn.norm_cross")
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # print(encoder_hidden_states.shape)
        key = attn.to_k(encoder_hidden_states)
        # print(key.shape)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query1.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        ##customization style transfer
        if self.style_matrix1 == None :
            from dct_util import dct, idct, low_pass, high_pass

            ###################################
            # --- 频域融合（原始输入） ---
            ###################################
            value_style = self.to_v_ip(ip_hidden_states_style).float()
            value_style_dct = dct(value_style, norm='ortho')

            value_content = self.to_v_ip(ip_hidden_states).float()
            value_content_dct = dct(value_content, norm='ortho')

            # 高频来自style，低频来自content
            merged_dct_mid = low_pass(value_style_dct, 0.15)\
                             + high_pass(low_pass(value_content_dct, 0.9), 0.15) \
                             + high_pass(value_style_dct, 0.9)

            self.style_matrix = idct(merged_dct_mid, norm='ortho')

            merged_dct_low = low_pass(value_style_dct, 0.45)\
                             + high_pass(low_pass(value_content_dct, 0.55), 0.45) \
                             + high_pass(value_style_dct, 0.55)
            self.style_matrix2 = idct(merged_dct_low, norm='ortho')

            ###################################
            # --- 奇异值调制 + 频域融合 ---
            ###################################
            weight_k = self.to_k_ip.weight.float()
            U_k, S_k, Vh_k = torch.linalg.svd(weight_k, full_matrices=False)

            # S_mod = 1 / S_k
            lambda_ = 0
            S_mod = 1 / torch.sqrt(S_k ** 2 + lambda_)
            # print(S_mod)
            transform_style = Vh_k.T @ torch.diag_embed(S_mod).T @ U_k.T
            style_proj = ip_hidden_states_style.float() @ transform_style
            style_proj_dct = dct(style_proj, norm='ortho')

            S_mod1 = S_k
            transform_content = Vh_k.T @ torch.diag_embed(S_mod1).T @ U_k.T
            content_proj = ip_hidden_states.float() @ transform_content
            content_proj_dct = dct(content_proj, norm='ortho')
            # print(content_proj.shape,content_proj_dct.shape)
            merged_dct_mid = low_pass(style_proj_dct, 0.15) \
                             + high_pass(low_pass(content_proj_dct, 0.9), 0.15) \
                             + high_pass(style_proj_dct, 0.9)
            self.style_matrix1 = idct(merged_dct_mid, norm='ortho')

            merged_dct_low_svd = low_pass(style_proj_dct, 0.45) \
                             + high_pass(low_pass(content_proj_dct, 0.55), 0.45) \
                             + high_pass(style_proj_dct, 0.55)
            self.style_matrix3 = idct(merged_dct_low_svd, norm='ortho')

            embeddings = F.softmax((self.style_matrix1), dim=-1)  # 转为概率分布
            eps = 1e-3  # 避免 log(0)
            entropy1 = -torch.sum(embeddings * torch.log(embeddings + eps), dim=-1)

            embeddings = F.softmax((self.style_matrix3), dim=-1)  # 转为概率分布
            entropy2 = -torch.sum(embeddings * torch.log(embeddings + eps), dim=-1)
            # print(entropy0.sum(),entropy1.sum(),entropy2.sum())
            scale_entropy = (entropy1.sum() / (entropy1.sum() + entropy2.sum()))
            self.scale_entropy = (self.scale_entropy * self.layer + scale_entropy) / (self.layer + 1)
            self.layer += 1

        ##color style transfer
        # if self.style_matrix1 == None :
        #     from dct_util import dct, idct, low_pass, high_pass
        #
        #     ###################################
        #     # --- 频域融合（原始输入） ---
        #     ###################################
        #     value_style = self.to_v_ip(ip_hidden_states_style).float()
        #     value_style_dct = dct(value_style, norm='ortho')
        #
        #     value_content = self.to_v_ip(ip_hidden_states).float()
        #     value_content_dct = dct(value_content, norm='ortho')
        #
        #     # 高频来自style，低频来自content
        #     merged_dct_high = low_pass(value_content_dct, 0.8) + high_pass(value_style_dct, 0.8)
        #     self.style_matrix = idct(merged_dct_high, norm='ortho')
        #
        #     # 低频来自style，高频来自content（反向组合）
        #     merged_dct_low = high_pass(value_style_dct, 0.05) + low_pass(value_content_dct, 0.05)
        #     self.style_matrix2 = idct(merged_dct_low, norm='ortho')
        #
        #     ###################################
        #     # --- 奇异值调制 + 频域融合 ---
        #     ###################################
        #     weight_k = self.to_k_ip.weight.float()
        #     U_k, S_k, Vh_k = torch.linalg.svd(weight_k, full_matrices=False)
        #
        #     lambda_ = 0#.01
        #     S_mod = 1 / torch.sqrt(S_k ** 2 + lambda_)
        #     # S_mod = 1 / S_k
        #     transform_style = Vh_k.T @ torch.diag_embed(S_mod).T @ U_k.T
        #     style_proj = ip_hidden_states_style.float() @ transform_style
        #     style_proj_dct = dct(style_proj, norm='ortho')
        #
        #     lambda_ = 0
        #     S_mod = 1 / torch.sqrt(S_k ** 2 + lambda_)
        #     S_mod1 = (S_k + S_mod) / 2# * 3 / 4
        #     # print(S_mod1)
        #     transform_content = Vh_k.T @ torch.diag_embed(S_mod1).T @ U_k.T
        #     content_proj = ip_hidden_states.float() @ transform_content
        #     content_proj_dct = dct(content_proj, norm='ortho')
        #
        #     # 高频来自style，低频来自content
        #     merged_dct_high_svd = low_pass(content_proj_dct, 0.8) + high_pass(style_proj_dct, 0.8)
        #     self.style_matrix1 = idct(merged_dct_high_svd, norm='ortho')
        #
        #     # 低频来自style，高频来自content
        #     merged_dct_low_svd = high_pass(style_proj_dct, 0.05) + low_pass(content_proj_dct, 0.05)
        #     self.style_matrix3 = idct(merged_dct_low_svd, norm='ortho')
        #
        #
        #     embeddings = F.softmax((self.style_matrix1), dim=-1)  # 转为概率分布
        #     eps = 1e-3  # 避免 log(0)
        #     entropy1 = -torch.sum(embeddings * torch.log(embeddings + eps), dim=-1)
        #
        #     embeddings = F.softmax((self.style_matrix3), dim=-1)  # 转为概率分布
        #     entropy2 = -torch.sum(embeddings * torch.log(embeddings + eps), dim=-1)
        #     # print(entropy0.sum(),entropy1.sum(),entropy2.sum())
        #     scale_entropy = (entropy1.sum() / (entropy1.sum() + entropy2.sum()))
        #     self.scale_entropy = (self.scale_entropy * self.layer + scale_entropy) / (self.layer + 1)
        #     self.layer += 1

        if self.denoise_step < 30 * self.scale_entropy:
            ip_key = self.style_matrix1.to(hidden_states.dtype)
            ip_value = self.style_matrix.to(hidden_states.dtype)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden_states1 = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states1 = ip_hidden_states1.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states1 = ip_hidden_states1.to(query.dtype)
            hidden_states += ip_hidden_states1 * 0.6
        else:
            ip_key = self.style_matrix3.to(hidden_states.dtype)
            ip_value = self.style_matrix2.to(hidden_states.dtype)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden_states1 = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states1 = ip_hidden_states1.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states1 = ip_hidden_states1.to(query.dtype)
            hidden_states += ip_hidden_states1 * 0.65

        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs,):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
