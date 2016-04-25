import Darwin
import Accelerate

extension Tensor {
    public var softmax: Tensor {
        let exps = exp
        let sum = exps.elements.reduce(0.0, combine: +)
        return exps / sum
    }
    
    public var relu: Tensor {
        return Tensor(shape: shape, elements: elements.map { fmax($0, 0.0) })
    }
}

extension Tensor {
    public func maxPool(ksize ksize: [Int], strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(shape.dimensions.count)")
        assert(ksize.count == 3, "`ksize.count` must be 3: \(ksize.count)")
        assert(ksize[2] == 1 ,"`ksize[3]` != 1 is not supported: \(ksize[2])")
        assert(strides.count == 3, "`strides.count` must be 3: \(strides.count)")
        assert(strides[2] == 1 ,"`strides[2]` != 1 is not supported: \(strides[2])")
        
        let numRows = shape.dimensions[0].value
        let numCols = shape.dimensions[1].value
        let numChannels = shape.dimensions[2].value
        
        let filterHeight = ksize[0]
        let filterWidth = ksize[1]
        
        let minDy = -(filterHeight - 1) / 2
        let maxDy = minDy + filterHeight - 1
        let minDx = -(filterWidth - 1) / 2
        let maxDx = minDx + filterWidth - 1
        
        let rowStride = strides[0]
        let colStride = strides[1]

        let outRows = shape.dimensions[0].value.ceilDiv(rowStride)
        let outCols = shape.dimensions[1].value.ceilDiv(colStride)
        
        // Initialize with -infinity for maximization.
        let elements = [Element](count: outCols * outRows * numChannels, repeatedValue: -Float.infinity)
        
        for y in 0..<outRows {
            let inY = y * rowStride
            let minY2 = max(inY + minDy, 0)
            let maxY2 = min(inY + maxDy, numRows - 1)

            for y2 in minY2...maxY2 {
                var outPixelIndex = y * outCols
                for x in 0..<outCols {
                    let inX = x * colStride
                    let minX2 = max(inX + minDx, 0)
                    let maxX2 = min(inX + maxDx, numCols - 1)
                    
                    var inPointer = UnsafeMutablePointer<Element>(self.elements) + (y2 * numCols + minX2) * numChannels
                    for _ in minX2...maxX2 {
                        var outPointer = UnsafeMutablePointer<Element>(elements) + outPixelIndex * numChannels
                        for _ in 0..<numChannels {
                            outPointer.memory = max(outPointer.memory, inPointer.memory)
                            outPointer += 1
                            inPointer += 1
                        }
                    }
                    outPixelIndex += 1
                }
            }
        }
        
        return Tensor(shape: [1, Dimension(outRows), Dimension(outCols), Dimension(numChannels)], elements: elements)
    }
    
    
    public func conv2d(filter filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        let inChannels = filter.shape.dimensions[2].value
        
        assert(shape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count == 3, "`strides.count` must be 3: \(strides.count)")
        assert(strides[2] == 1, "`strides[2]` must be 1")
        assert(shape.dimensions[2].value == inChannels, "The number of channels of this tensor and the filter are not compatible: \(shape.dimensions[2]) != \(inChannels)")
        
        let inRows = shape.dimensions[0].value
        let inCols = shape.dimensions[1].value
        
        let filterHeight = filter.shape.dimensions[0].value
        let filterWidth = filter.shape.dimensions[1].value
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = strides[0]
        let colStride = strides[1]
        
        let outRows = inRows.ceilDiv(rowStride)
        let outCols = inCols.ceilDiv(colStride)
        let outChannels = filter.shape.dimensions[3].value
        
        let elementsPointer = UnsafePointer<Float>(elements)
        
        // a.shape == [outRows * outCols, rowSize]
        let rowSize = filterHeight * filterWidth * inChannels
        let a = [Float](count: outRows * outCols * rowSize, repeatedValue: 0)
        for y in 0..<outRows {
            let inY = y * rowStride
            let inMinY = max(inY + inMinDy, 0)
            let inMaxY = min(inY + inMaxDy, inRows - 1)
            
            for x in 0..<outCols{
                let inX = x * colStride
                let inMinX = max(inX + inMinDx, 0)
                let inMaxX = min(inX + inMaxDx, inCols - 1)
                
                // Add (x,y)'s patch as a vector
                var dest = UnsafeMutablePointer<Float>(a) + ((y * outCols + x) * filterHeight - min(inY + inMinDy, 0)) * filterWidth * inChannels
                var src = elementsPointer + (inMinY * inCols + inMinX) * inChannels
                for _ in inMinY...inMaxY {
                    memcpy(dest - min(inMinX + inMinDx, 0) * inChannels, src, (inMinX...inMaxX).count * inChannels * sizeof(Float))
                    dest += filterWidth * inChannels
                    src += inCols * inChannels
                }
            }
        }
        
        let result = Tensor(shape: [Dimension(outRows), Dimension(outCols), Dimension(outChannels)])
        
        let n = Int32(outChannels)
        let k = Int32(rowSize)
        
        // Calculate [M, N] matrix, it automatically turns into [outRows, outCols, outChannels] Tensor
        cblas_sgemm(
            CblasRowMajor,                                // Order
            CblasNoTrans,                                 // TransA
            CblasNoTrans,                                 // TransB
            Int32(outRows * outCols),                     // M
            n,                                            // N
            k,                                            // K
            1.0,                                          // alpha
            UnsafePointer<Float>(a),                      // A
            k,                                            // lda
            UnsafePointer<Float>(filter.elements),        // B
            n,                                            // ldb
            1.0,                                          // beta
            UnsafeMutablePointer<Float>(result.elements), // C
            n
        )
        
        return result
    }
}