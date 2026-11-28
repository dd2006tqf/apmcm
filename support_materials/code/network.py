"""Flood-cut road network shared by Q4 (access time, road-cut share) and Q5 (travel times).

OSM roads -> directed graph by 1 m vertex snapping (oneway=yes respected); edge time = length / speed
(A13 x SPEED_FACTOR). An edge is impassable if any sample point (every 20 m, A14) lies within
ROAD_BUFFER_M of the new-inundation mask F. Shortest paths by Dijkstra (scipy.sparse.csgraph).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import config as C


def speed_of(hw):
    hw = str(hw)
    if hw.endswith("_link"):
        return max(20.0, C.SPEEDS_KMH.get(hw[:-5], 20) - 20)
    return float(C.SPEEDS_KMH.get(hw, 20))


class RoadNet:
    def __init__(self, crs):
        roads = gpd.read_file(C.PUB / "OSM" / "roads.geojson").to_crs(crs)
        roads = roads.explode(index_parts=False)
        key, xy = {}, []
        eu, ev, elen, espd, eone, emid, segs = [], [], [], [], [], [], []
        def node(x, y):
            k = (round(x), round(y))
            if k not in key:
                key[k] = len(xy)
                xy.append(k)
            return key[k]
        for hw, ow, geom in zip(roads["highway"], roads.get("oneway"), roads.geometry):
            if geom is None or geom.is_empty:
                continue
            cs = np.asarray(geom.coords)[:, :2]
            sp = speed_of(hw)
            one = str(ow) == "yes"
            for (x0, y0), (x1, y1) in zip(cs[:-1], cs[1:]):
                u, v = node(x0, y0), node(x1, y1)
                if u == v:
                    continue
                L = float(np.hypot(x1 - x0, y1 - y0))
                eu.append(u); ev.append(v); elen.append(L); espd.append(sp); eone.append(one)
                segs.append((x0, y0, x1, y1))
        self.xy = np.array(xy, float)
        self.eu, self.ev = np.array(eu), np.array(ev)
        self.len_m = np.array(elen)
        self.speed = np.array(espd)
        self.oneway = np.array(eone)
        self.segs = np.array(segs)
        self.n = len(self.xy)

    def cut_mask(self, F, transform, buffer_m=C.ROAD_BUFFER_M, step=C.ROAD_SAMPLE_M):
        """Boolean per edge: True = impassable (a sample point within buffer_m of F)."""
        res = transform.a
        rad = int(np.ceil(buffer_m / res))
        offs = [(dr, dc) for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1) if (dr * dr + dc * dc) * res * res <= buffer_m ** 2 + 1e-9]
        nseg = np.maximum(1, np.ceil(self.len_m / step).astype(int))
        sid = np.repeat(np.arange(len(nseg)), nseg + 1)
        t = np.concatenate([np.linspace(0, 1, n + 1) for n in nseg])
        x = self.segs[sid, 0] + t * (self.segs[sid, 2] - self.segs[sid, 0])
        y = self.segs[sid, 1] + t * (self.segs[sid, 3] - self.segs[sid, 1])
        cols = np.floor((x - transform.c) / transform.a).astype(int)
        rows = np.floor((y - transform.f) / transform.e).astype(int)
        H, W = F.shape
        hit = np.zeros(len(x), bool)
        for dr, dc in offs:
            rr, cc = rows + dr, cols + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            hit[ok] |= np.asarray(F[rr[ok], cc[ok]]).astype(bool)
        return np.bincount(sid, weights=hit, minlength=len(nseg)) > 0

    def graph(self, cut=None, speed_factor=C.SPEED_FACTOR):
        keep = np.ones(len(self.eu), bool) if cut is None else ~cut
        tmin = self.len_m / (self.speed * speed_factor * 1000 / 60)       # minutes
        u, v, w = self.eu[keep], self.ev[keep], tmin[keep]
        two = ~self.oneway[keep]
        uu = np.concatenate([u, v[two]]); vv = np.concatenate([v, u[two]]); ww = np.concatenate([w, w[two]])
        # de-duplicate parallel edges keeping the minimum time (csr would sum duplicates)
        order = np.lexsort((ww, vv, uu))
        uu, vv, ww = uu[order], vv[order], ww[order]
        first = np.ones(len(uu), bool)
        first[1:] = (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1])
        G = csr_matrix((ww[first], (uu[first], vv[first])), shape=(self.n, self.n))
        active = np.zeros(self.n, bool)
        active[uu] = True; active[vv] = True
        return G, active

    def snap(self, pts, active):
        idx = np.where(active)[0]
        tree = cKDTree(self.xy[idx])
        d, j = tree.query(pts)
        return idx[j], d

    def times_from(self, G, sources, reverse=False):
        """Dijkstra travel time (min) from each source node to all nodes (reverse=True: to the source)."""
        Gd = G.T.tocsr() if reverse else G
        return dijkstra(Gd, directed=True, indices=np.atleast_1d(sources))
