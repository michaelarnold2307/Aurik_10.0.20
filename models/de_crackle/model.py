"""VinylNet v4.

Komponenten:
  ClickNet   gelernter Klickdetektor (dilatiertes TCN auf Welle + 1./2. Differenz
             plus M/S — gemeinsame Kanaleingabe fuer versetzte Stereo-Klicks)
             -> Klickwahrscheinlichkeit je Kanal
  InpaintNet gelerntes Luecken-Inpainting (U-Net, Zeitdomaene): ersetzt die
             AR-Interpolation bei grossen Luecken (v3-Schwachstelle), kleine
             Luecken bleiben klassisch/roh
  ThumpNet   eigener Pfad fuer tieffrequente Pops/Thumps (arbeitet auf 32x unterabgetastetem Signal)
  VinylNet   Stereo-Band-Split-Transformer: Zeit-Attention mit RoPE + Band-Attention,
             Eingabe: repariertes UND rohes Signal + Klickmasken-Dichte (Netz entscheidet, wo Reparatur taugt)
             -> komplexe Maske (startet als Identitaet) -> iSTFT -> stereo TCN-Refiner
  MRDisc     Multi-Resolution-STFT-Diskriminator (optional, GAN-Phase)

`strength` skaliert Thump-Abzug, Maske und Refiner (0 = Original, 1 = volle Restaurierung).

v4-Gegenueber-v3:
  * ClickNet sieht Mid/Side explizit (8 Eingabekanäle statt 6) — laterale vs.
    vertikale Rillenbewegung und links/rechts versetzte Klicks werden direkt kodiert
  * InpaintNet füllt grosse Lücken generativ (AR-LSAR bleibt nur noch Kurz-Lücken-Fallback)
  * neues Preset `xl` für die maximale Ausbaustufe (GPU, 200+ Epochen)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny": dict(n_fft=512, hop=128, dim=32, layers=1, heads=2, ref_ch=8, ref_layers=4, thump_ch=8, thump_layers=5),
    "small": dict(n_fft=2048, hop=512, dim=128, layers=4, heads=4, ref_ch=32, ref_layers=8, thump_ch=32, thump_layers=9),
    "base": dict(n_fft=2048, hop=512, dim=192, layers=6, heads=6, ref_ch=48, ref_layers=8, thump_ch=32, thump_layers=9),
    "large": dict(n_fft=2048, hop=512, dim=256, layers=8, heads=8, ref_ch=64, ref_layers=10, thump_ch=48, thump_layers=9),
    "xl": dict(n_fft=2048, hop=512, dim=320, layers=10, heads=10, ref_ch=80, ref_layers=12, thump_ch=64, thump_layers=11),
}
FEAT_CH = 8  # [Re,Im] x [L,R] x [repariert, roh]
AUX_CH = 2  # Klickmasken-Dichte L,R


# ------------------------------------------------------------------ Klickdetektor
class ChanNorm(nn.Module):
    """LayerNorm ueber die Kanaele (unabhaengig von der Clip-Laenge)."""

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):  # (B,C,T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ClickNet(nn.Module):
    """Gelernter Klickdetektor.

    Eingabe: 8 asinh-komprimierte Kanäle [xL, xR, dxL, dxR, d2L, d2R, M, S] —
    die gemeinsame Kanaleingabe (Conv1d ueber alle 8) macht links/rechts versetzte
    Klicks und Phasenlage (vertikale Rillenbewegung) direkt sichtbar.
    """

    def __init__(self, ch=64, layers=12, feat_ch=8):
        super().__init__()
        dils = [2 ** (i % 8) for i in range(layers)]
        self.inp = nn.Conv1d(feat_ch, ch, 5, padding=2)
        self.norms = nn.ModuleList([ChanNorm(ch) for _ in dils])
        self.convs = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = nn.Conv1d(ch, 2, 1)

    @staticmethod
    def features(x):  # (B,2,T) -> (B,8,T), asinh-komprimiert (grosser Dynamikbereich) + M/S
        dx = F.pad(x[..., 1:] - x[..., :-1], (1, 0))
        d2 = F.pad(dx[..., 1:] - dx[..., :-1], (1, 0))
        s2 = torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))
        m = ((x[:, 0] + x[:, 1]) / s2).unsqueeze(1)
        sd = ((x[:, 0] - x[:, 1]) / s2).unsqueeze(1)
        return torch.asinh(100 * torch.cat([x, dx, d2, m, sd], 1))

    def forward(self, x):  # -> Logits (B,2,T)
        h = self.inp(self.features(x))
        for n, c in zip(self.norms, self.convs):
            h = h + c(F.gelu(n(h)))
        return self.out(h)


# ------------------------------------------------------------------ Inpainting (v4 neu)
class _EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class _DecBlock(nn.Module):
    """Upsample + Skip-Konkatenation + Kanal-Ausgleich."""

    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.up = nn.ConvTranspose1d(cin, cout, 2, stride=2)
        self.conv = nn.Conv1d(cout + skip_ch, cout, 3, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x, skip):
        t = min(x.shape[-1] * 2, skip.shape[-1])
        x = F.gelu(self.up(x))[:, :, :t]
        return F.gelu(self.norm(self.conv(torch.cat([x, skip[:, :, -t:]], 1))))


class InpaintNet(nn.Module):
    """Gelerntes Luecken-Inpainting (v4).

    Eingabe: Signal mit genullten Luecken + Maske -> Vollsignal-Schaetzung.
    Residual-Form: out = inp + net(inp, mask); die Luecke wird durch das Netz gefuellt,
    der Kontext bleibt (nahezu) erhalten. Fuer grosse Luecken ersetzt es die AR-Interpolation
    (deren Fehler ueberlaenger Luecken den Klick selbst ubersteigt — v3-Befund).
    """

    def __init__(self, ch=24):
        super().__init__()
        self.e1 = _EncBlock(3, ch)  # [L_genullt, R_genullt, Maske]
        self.e2 = _EncBlock(ch, 2 * ch)
        self.e3 = _EncBlock(2 * ch, 4 * ch)
        self.mid = nn.ModuleList([nn.Conv1d(4 * ch, 4 * ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(6)])
        self.d3 = _DecBlock(4 * ch, 2 * ch, 2 * ch)
        self.d2 = _DecBlock(2 * ch, ch, ch)
        self.out = nn.Linear(ch, 2)

    def forward(self, x_zeroed, mask):  # (B,2,T), (B,1,T) 0/1 -> (B,2,T)
        m3 = torch.cat([x_zeroed, mask], 1)
        h1 = self.e1(m3)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        for l in self.mid:
            h3 = h3 + F.gelu(l(h3))
        d3 = self.d3(h3, h2)
        y = F.gelu(self.d2(d3, h1))
        y = F.interpolate(y, size=x_zeroed.shape[-1], mode="linear", align_corners=False)
        return x_zeroed + self.out(y.permute(0, 2, 1)).permute(0, 2, 1)


# ------------------------------------------------------------------ Thump-Pfad
class ThumpNet(nn.Module):
    def __init__(self, ch=32, layers=9, down=32):
        super().__init__()
        self.down = down
        self.inp = nn.Conv1d(2, ch, 3, padding=1)
        self.tcn = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
        self.out = nn.Conv1d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):  # (B,2,T) -> geschaetzter Thump (B,2,T)
        B, C, T = x.shape
        Tp = -(-T // self.down) * self.down
        xp = F.pad(x, (0, Tp - T), mode="replicate")
        h = F.gelu(self.inp(F.avg_pool1d(xp, self.down, self.down) * 4))
        for l in self.tcn:
            h = h + F.gelu(l(h))
        y = self.out(h) / 4
        return F.interpolate(y, size=Tp, mode="linear", align_corners=False)[..., :T]


# ------------------------------------------------------------------ Restaurierungsnetz
def make_bands(n_bins, base=4, per_group=8):
    edges, w = [0], base
    while edges[-1] < n_bins:
        for _ in range(per_group):
            if edges[-1] >= n_bins:
                break
            edges.append(min(edges[-1] + w, n_bins))
        w *= 2
    return edges


class BandSplit(nn.Module):
    def __init__(self, edges, dim):
        super().__init__()
        self.edges = edges
        ws = [b - a for a, b in zip(edges[:-1], edges[1:])]
        self.norms = nn.ModuleList([nn.LayerNorm(FEAT_CH * w) for w in ws])
        self.proj = nn.ModuleList([nn.Linear(FEAT_CH * w + 1 + AUX_CH, dim) for w in ws])
        self.band_emb = nn.Parameter(torch.randn(1, len(ws), 1, dim) * 0.02)

    def forward(self, feat, aux):  # feat (B,8,F,N), aux (B,2,N) -> (B,bands,N,dim)
        B, _, _, N = feat.shape
        a = aux.transpose(1, 2).to(feat.dtype)
        outs = []
        for i, (f0, f1) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            z = feat[:, :, f0:f1, :].permute(0, 3, 1, 2).reshape(B, N, FEAT_CH * (f1 - f0))
            e = torch.log(z.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(z.dtype)  # Absolutpegel
            outs.append(self.proj[i](torch.cat([self.norms[i](z), e, a], -1)))
        return torch.stack(outs, 1) + self.band_emb


class MaskHead(nn.Module):
    def __init__(self, edges, dim):
        super().__init__()
        ws = [b - a for a, b in zip(edges[:-1], edges[1:])]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 4 * w)) for w in ws
        ])
        for h in self.heads:  # Start = Identitaet
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, h):  # (B,bands,N,dim) -> (B,4,F,N)
        outs = []
        for i, head in enumerate(self.heads):
            o = head(h[:, i])
            B, N, _ = o.shape
            outs.append(o.reshape(B, N, 4, -1).permute(0, 2, 3, 1))
        return torch.cat(outs, 2)


def _rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class RopeBlock(nn.Module):
    """Pre-Norm-Transformerblock mit Rotary Position Embeddings (relative Positionen, laengenrobust)."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        self.heads, self.hd = heads, dim // heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):  # (B,N,D)
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        f = torch.outer(torch.arange(N, device=x.device, dtype=torch.float32), inv)
        cos, sin = f.cos().to(q.dtype), f.sin().to(q.dtype)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(o)
        return x + self.mlp(self.n2(x))


class AxialBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.t_norm = nn.LayerNorm(dim)
        self.t_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)  # lokale Zeitstruktur
        self.t_attn = RopeBlock(dim, heads)
        self.b_attn = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.0, "gelu", batch_first=True, norm_first=True)

    def forward(self, x):  # (B,bands,N,D)
        B, Bd, N, D = x.shape
        t = x.reshape(B * Bd, N, D)
        t = t + self.t_conv(self.t_norm(t).transpose(1, 2)).transpose(1, 2)
        x = self.t_attn(t).reshape(B, Bd, N, D)
        b = x.permute(0, 2, 1, 3).reshape(B * N, Bd, D)
        return self.b_attn(b).reshape(B, N, Bd, D).permute(0, 2, 1, 3)


class Refiner(nn.Module):
    def __init__(self, ch=32, layers=8):
        super().__init__()
        self.inp = nn.Conv1d(4, ch, 3, padding=1)
        self.tcn = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
        self.out = nn.Conv1d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, est, ref):
        h = F.gelu(self.inp(torch.cat([est, ref], 1)))
        for l in self.tcn:
            h = h + F.gelu(l(h))
        return est + self.out(h)


class VinylNet(nn.Module):
    """Restaurierungsnetz. In v4 um `inpaint` (InpaintNet) ergaenzt — wird im selben
    Training mitgelernt und zur Inferenz fuer grosse Luecken verwendet."""

    def __init__(self, n_fft=2048, hop=512, dim=128, layers=4, heads=4, ref_ch=32, ref_layers=8,
                 thump_ch=32, thump_layers=9, inpaint_ch=24):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        edges = make_bands(n_fft // 2 + 1)
        self.thump = ThumpNet(thump_ch, thump_layers)
        self.split = BandSplit(edges, dim)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads) for _ in range(layers)])
        self.head = MaskHead(edges, dim)
        self.refiner = Refiner(ref_ch, ref_layers)
        self.inpaint = InpaintNet(inpaint_ch)
        self.register_buffer("win", torch.hann_window(n_fft))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _stft(self, x):
        B, C, T = x.shape
        X = torch.stft(x.float().reshape(B * C, T), self.n_fft, self.hop, window=self.win, return_complex=True)
        return X.reshape(B, C, *X.shape[1:])

    def inpaint_fill(self, noisy, mask):  # (B,2,T), (B,2,T) 0/1 -> gefuelltes Signal
        mz = mask.float().mean(1, keepdim=True).clamp(0.0, 1.0)
        z = noisy * (1 - mask.float())
        return self.inpaint(z, mz)

    def forward(self, x, strength=1.0, click_mask=None, x_raw=None):
        """x: repariertes (oder rohes) Signal (B,2,T). x_raw: unrepariertes Original (Default: x).
        click_mask: (B,2,T) 0/1 Detektor-Maske (Default: keine)."""
        B, C, T = x.shape
        rms = x.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        xn = x / rms
        rn = xn if x_raw is None else x_raw / rms
        x1 = xn - strength * self.thump(xn)
        dev = xn.device.type
        with torch.autocast(device_type=dev, enabled=False):
            X = self._stft(x1)
            R = self._stft(rn)
            feat = torch.stack([X[:, 0].real, X[:, 0].imag, X[:, 1].real, X[:, 1].imag,
                                R[:, 0].real, R[:, 0].imag, R[:, 1].real, R[:, 1].imag], 1)
            N = X.shape[-1]
            if click_mask is None:
                aux = torch.zeros(B, 2, N, device=x.device)
            else:
                aux = F.avg_pool1d(click_mask.float(), self.n_fft, self.hop, padding=self.n_fft // 2)[..., :N]
        h = self.split(feat, aux)
        for blk in self.blocks:
            h = blk(h)
        m = self.head(h)
        with torch.autocast(device_type=dev, enabled=False):
            m = m.float()
            M = 1 + strength * 1.5 * torch.tanh(torch.complex(m[:, 0::2], m[:, 1::2]))
            Y = (X * M).reshape(B * C, *X.shape[2:])
            est = torch.istft(Y, self.n_fft, self.hop, window=self.win, length=T).reshape(B, C, T)
        out = est + strength * (self.refiner(est, x1) - est)
        return out * rms


# ------------------------------------------------------------------ Diskriminator
class DiscR(nn.Module):
    def __init__(self, n_fft, ch):
        super().__init__()
        self.n_fft, self.hop = n_fft, n_fft // 4
        sn = nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            sn(nn.Conv2d(2, ch, (3, 9), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 9), stride=(2, 2), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 9), stride=(2, 2), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 3), padding=1)),
        ])
        self.out = sn(nn.Conv2d(ch, 1, (3, 3), padding=1))
        self.register_buffer("win", torch.hann_window(n_fft))

    def forward(self, x):
        B, C, T = x.shape
        S = torch.stft(x.reshape(B * C, T), self.n_fft, self.hop, window=self.win, return_complex=True).abs()
        h = torch.log1p(S.reshape(B, C, *S.shape[1:]) * 10)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        return self.out(h), feats


class MRDisc(nn.Module):
    def __init__(self, ch=32, ffts=(512, 1024, 2048)):
        super().__init__()
        self.ds = nn.ModuleList([DiscR(n, ch) for n in ffts])

    def forward(self, x):
        return [d(x) for d in self.ds]


def build_vinylnet(preset: str, **overrides) -> VinylNet:
    """VinylNet aus Presets-Name bauen (konfiguriertes Checkpoint-Laden)."""
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return VinylNet(**cfg)


def build_clicknet(ch=64, layers=12, feat_ch=8) -> ClickNet:
    return ClickNet(ch=ch, layers=layers, feat_ch=feat_ch)
