import prody

def calc_mat_val(a1, a2, xyz1, xyz2, q1, q2, r1, r2, p1, p2, parms):
    dmat = min(parms["distmat_cutoff"], prody.measure.measure.getDistance(xyz1, xyz2))
    if a1==a2:
        cmat = q1
    else:
        cmat = parms["f"]*(q1*q2) / (parms["e_r"]*dmat)
        cmat = min(parms["cmaxcut"],max(parms["cmincut"], cmat ))
    sdmat = min(parms["max_seq_dist"], abs(r1-r2))
    simat = min(r1, r2)%2 * min(parms["max_seq_dist"], abs(r1-r2))
    if parms["do_pocket"]:
        pmat = p1 + p2
        
    else:
        pmat = 0
    
    return (dmat, cmat, sdmat, simat, pmat)

def data_stream_val(N, xyz, q, r, p, parms):
    #print(N, xyz, q, r, p, parms)
    for a1 in N:
        xyz1 = xyz[a1]
        q1 = q[a1]
        r1 = r[a1]
        p1 = p[a1]

        for a2 in N:
            xyz2 = xyz[a2]
            q2 = q[a2]
            r2 = r[a2]
            p2 = p[a2]
            if a1 <= a2:
                yield a1, a2, xyz1, xyz2, q1, q2, r1, r2, p1, p2, parms

def proxy(args):
    return args[:2], calc_mat_val(*args)