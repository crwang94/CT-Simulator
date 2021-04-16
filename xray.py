import multiprocessing
from multiprocessing import Pool
import os
import numpy as np
import math
from stl import mesh
from PIL import Image
from itertools import product


def attachHits(structure):
    tri = structure[0]
    triY = tri[:, 1]

    matrix = structure[1]
    result = []

    triProjected = np.delete(tri, 1, 1)
    minx = np.ceil(min(triProjected[:, 0])).astype(int)
    maxx = np.ceil(max(triProjected[:, 0])).astype(int)
    minz = np.ceil(min(triProjected[:, 1])).astype(int)
    maxz = np.ceil(max(triProjected[:, 1])).astype(int)

    temp = list(product(range(minx, maxx), range(minz, maxz)))

    if (len(temp) != 0):
        Ps = np.array(temp)
        PAs = Ps-triProjected[0]

        UVs = np.matmul(PAs, matrix)

        for item in zip(Ps, UVs):
            p = item[0]
            uv = item[1]
            if uv[0] > 0 and uv[1] > 0 and sum(uv) < 1:
                y = (1-sum(uv))*triY[0]+uv[0]*triY[1]+uv[1]*triY[2]
                pc = np.append(p, y)
                result.append(pc)
    # return [tri, result]

    return result


def createPixelStorage2(pixelss):

    pixelStore = {}
    for pixels in pixelss:
        for pixel in pixels:
            p = (int(round(pixel[0])), int(round(pixel[1])))
            if not p in pixelStore:
                pixelStore[p] = []
            pixelStore[p].append(pixel[2])
    return pixelStore


def sortAndCompute(ls):
    sortedLs = sorted(ls)
    o = sortedLs[1::2]
    e = sortedLs[0::2]
    return sum(o)-sum(e)


def extractMatrices(triangles):

    BAs = np.subtract(triangles[:, 1, :], triangles[:, 0, :])
    CAs = np.subtract(triangles[:, 2, :], triangles[:, 0, :])
    result = np.stack((BAs, CAs), axis=1)
    return np.delete(result, 1, 2)


def initMesh(name, resolutionHalf):

    mmesh = mesh.Mesh.from_file(name)
    triangles = mmesh.vectors
    _, cog, _ = mmesh.get_mass_properties()

    for triangle in triangles:
        for i in range(3):
            triangle[:, i] -= cog[i]

    maxWidthX = max(abs(mmesh.x.min()), abs(mmesh.x.max()))
    maxWidthY = max(abs(mmesh.y.min()), abs(mmesh.y.max()))
    maxWidthZ = max(abs(mmesh.z.min()), abs(mmesh.z.max()))

    maxLen = np.ceil(np.sqrt(maxWidthX**2 + maxWidthY **
                             2 + maxWidthZ**2)).astype(int)

    scaleFactor = resolutionHalf/maxLen

    for triangle in triangles:
        for i in range(3):
            triangle[:, i] *= scaleFactor

    mmesh.rotate([0.5, 0.0, 0.0], math.radians(-91))
    mmesh.rotate([0.0, 0.0, 0.5], math.radians(90.33))
    mmesh.update_normals()
    return mmesh


if __name__ == "__main__":

    numImages = 360
    stepAng = 360/numImages

    name = 'head.stl'
    resolutionHalf = 500
    mmesh = initMesh(name, resolutionHalf)
    triangles = mmesh.vectors

    outputDir = os.path.splitext(name)[0]
    os.mkdir(outputDir)

    maxv = 0
    results = []

    for i in range(numImages):
        print("Computing image ", i)
        mmesh.rotate([1, 0, 0], math.radians(stepAng))

        matrices = np.linalg.inv(extractMatrices(triangles))

        structures = zip(triangles, matrices)

        pool = Pool()

        TriPixel = pool.map(attachHits, structures)

        m = createPixelStorage2(TriPixel)

        dist = {k: sortAndCompute(v) for k, v in m.items()}

        canvas = np.zeros((resolutionHalf*2, resolutionHalf*2))

        #maxv = 0
        for k, v in dist.items():
            canvas[k[0]+resolutionHalf, k[1]+resolutionHalf] = v
            if v > maxv:
                maxv = v

        results.append(canvas)

    rho = 1/maxv

    print("generating images...")
    for i in range(numImages):
        canvas = results[i]
        #canvas *= 255/maxv

        def f(x): return 255 * (1 - np.exp(-x*rho))
        canvas = f(canvas)

        #canvas *= np.exp()

        im = Image.fromarray(canvas)
        im = im.convert('L')
        im.save(outputDir + '/outfile'+str(i) + '.png')
