#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import torch

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, store_prof, write_log
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator, model_opt = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        if i == 0 and opt.trace:
            with torch.autograd.profiler.profile(record_shapes=True) as prof:
                _, _, stat = translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=opt.src_dir,
                    batch_size=opt.batch_size,
                    batch_type=opt.batch_type,
                    translate_steps=opt.translate_steps,
                    attn_debug=opt.attn_debug,
                    align_debug=opt.align_debug
                )
            store_prof(prof,
                       model_opt.encoder_type + '_' + model_opt.decoder_type,
                       model_opt.model_type, opt.batch_size, train=False)
        else:
            _, _, stat = translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                translate_steps=opt.translate_steps,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug
            )

        write_log(model_opt.encoder_type + '_' + model_opt.decoder_type,
                  model_opt.model_type,
                  stat.batch_avg_time,
                  stat.token_per_sec,
                  opt.batch_size,
                  opt.beam_size,
                  train=False)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()

    translate(opt)


if __name__ == "__main__":
    main()
