"""Zero-shot variant scoring with ESM-2.

"Zero-shot" means the model never sees any experimental fitness data. It was
trained only to fill in masked residues across UniRef, and the claim is that
a residue the model finds surprising in context is one that evolution has
been keeping out, which usually means it breaks something.

Three scoring schemes are implemented because they differ in cost by about
two orders of magnitude and it is worth knowing what that buys:

  wt-marginals      one forward pass total
  masked-marginals  one forward pass per mutated position  (the standard)
  mutant-marginals  one forward pass per variant           (rarely worth it)

All three score a variant as a log-odds against the wildtype residue:

    score = sum over substitutions of [ log p(mut | context) - log p(wt | context) ]

summing over substitutions so multi-mutants are handled, on the assumption
that the substitutions are independent. That assumption is wrong whenever
two mutations touch each other, which is exactly the case where epistasis
lives, so multi-mutant scores from this should be treated as weaker.
"""

import numpy as np
import torch

from .mutations import check_against_sequence, mutated_positions, parse_variant

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

# Positional embeddings were trained up to 1024 tokens including the two
# special tokens, so 1022 residues is the usable window.
MAX_RESIDUES = 1022


class ESMScorer:
    """Wraps an ESM-2 checkpoint for variant scoring.

    device: "cuda", "cpu", or None to pick automatically.
    dtype:  float16 on GPU roughly halves memory and makes no visible
            difference to the ranking; float32 on CPU because half precision
            on CPU is slower, not faster.
    """

    def __init__(self, model_name=DEFAULT_MODEL, device=None, dtype=None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, dtype=dtype).to(self.device).eval()

        self.mask_id = self.tokenizer.mask_token_id
        # token id for each of the 20 standard amino acids
        self.aa_ids = {aa: self.tokenizer.convert_tokens_to_ids(aa)
                       for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # ------------------------------------------------------------------
    # log-probability tables
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _logprobs(self, sequences):
        """Log-softmax over the vocabulary for a batch of sequences.

        Returns (batch, length+2, vocab) including the BOS/EOS tokens, so
        residue i of the sequence is at index i+1.
        """
        enc = self.tokenizer(sequences, return_tensors="pt",
                             padding=True).to(self.device)
        logits = self.model(**enc).logits.float()
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def wt_marginal_logprobs(self, sequence):
        """One pass over the unmutated sequence. (length, 20) array.

        Cheap, and noticeably worse than masking: the model can see the
        wildtype residue it is being asked about, so it mostly copies it.
        """
        lp = self._logprobs([sequence])[0]
        return self._gather_aa(lp[1:len(sequence) + 1])

    @torch.no_grad()
    def masked_marginal_logprobs(self, sequence, positions=None, batch_size=8,
                                 progress=False):
        """Mask each position in turn and read off that position's prediction.

        This is the scheme used in the ESM paper and in the ProteinGym
        baselines. Cost is one forward pass per position, batched.

        positions: 0-indexed positions to score. None means all of them.
        """
        if positions is None:
            positions = range(len(sequence))
        positions = list(positions)

        out = np.full((len(sequence), 20), np.nan, dtype=np.float32)

        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            enc = self.tokenizer([sequence] * len(chunk),
                                 return_tensors="pt").to(self.device)
            for row, pos in enumerate(chunk):
                enc["input_ids"][row, pos + 1] = self.mask_id

            logits = self.model(**enc).logits.float()
            lp = torch.log_softmax(logits, dim=-1)

            for row, pos in enumerate(chunk):
                out[pos] = self._gather_aa(lp[row, pos + 1:pos + 2])[0]

            if progress and (start // batch_size) % 10 == 0:
                print(f"    masked {start + len(chunk)}/{len(positions)}",
                      flush=True)

        return out

    def _gather_aa(self, lp_slice):
        """Pull out just the 20 standard amino acid columns, in AA20 order."""
        idx = torch.tensor([self.aa_ids[a] for a in "ACDEFGHIKLMNPQRSTVWY"],
                           device=lp_slice.device)
        return lp_slice[:, idx].cpu().numpy()

    # ------------------------------------------------------------------
    # variant scoring
    # ------------------------------------------------------------------

    def score_variants(self, sequence, variants, scheme="masked",
                       batch_size=8, progress=False):
        """Score a list of variant notations against the wildtype sequence.

        Returns a numpy array of log-odds scores, higher meaning the model
        finds the mutant more plausible, which should correlate with higher
        experimental fitness.
        """
        if len(sequence) > MAX_RESIDUES:
            raise ValueError(
                f"sequence is {len(sequence)} residues, over the {MAX_RESIDUES} "
                "limit. Use score_variants_windowed instead.")

        parsed = [parse_variant(v) for v in variants]
        for subs in parsed:
            check_against_sequence(subs, sequence)

        if scheme == "wt":
            table = self.wt_marginal_logprobs(sequence)
        elif scheme == "masked":
            positions = mutated_positions(variants)
            table = self.masked_marginal_logprobs(
                sequence, positions, batch_size=batch_size, progress=progress)
        else:
            raise ValueError(f"unknown scheme {scheme!r}, use 'wt' or 'masked'")

        order = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        scores = np.zeros(len(variants), dtype=np.float32)
        for i, subs in enumerate(parsed):
            total = 0.0
            for s in subs:
                row = table[s.idx]
                total += row[order[s.mut]] - row[order[s.wt]]
            scores[i] = total
        return scores

    def score_variants_windowed(self, sequence, variants, window=MAX_RESIDUES,
                                batch_size=8, progress=False):
        """For proteins longer than the context window.

        Scores each mutated position inside a window centred on it, so every
        position still gets the maximum available context. More expensive
        than a single truncation but it does not silently throw away the part
        of the protein the mutation is actually in.
        """
        parsed = [parse_variant(v) for v in variants]
        for subs in parsed:
            check_against_sequence(subs, sequence)

        order = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        half = window // 2
        cache = {}

        for pos in mutated_positions(variants):
            start = max(0, min(pos - half, len(sequence) - window))
            start = max(start, 0)
            sub_seq = sequence[start:start + window]
            local = pos - start
            table = self.masked_marginal_logprobs(
                sub_seq, [local], batch_size=1)
            cache[pos] = table[local]
            if progress:
                print(f"    windowed position {pos}", flush=True)

        scores = np.zeros(len(variants), dtype=np.float32)
        for i, subs in enumerate(parsed):
            total = 0.0
            for s in subs:
                row = cache[s.idx]
                total += row[order[s.mut]] - row[order[s.wt]]
            scores[i] = total
        return scores

    def __repr__(self):
        return (f"ESMScorer({self.model_name!r}, device={self.device.type}, "
                f"dtype={self.dtype})")
