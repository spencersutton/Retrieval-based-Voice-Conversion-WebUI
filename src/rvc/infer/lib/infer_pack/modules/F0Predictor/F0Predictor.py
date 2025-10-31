class F0Predictor:
    def compute_f0(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        """

    def compute_f0_uv(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
