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
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numOutChannels = filter.shape.dimensions[3].value
        
        let padAlongHeight = (numRows - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        let imageWidth = shape.dimensions[2].value
        let imageHeight = shape.dimensions[1].value
        let numInChannels = shape.dimensions[3].value
        
        let filterWidth = filter.shape.dimensions[1].value
        let filterHeight = filter.shape.dimensions[0].value
        
        
        let z = Tensor(shape: [Dimension(numBatches), Dimension(numRows), Dimension(numCols), Dimension(numOutChannels)])
        
//      https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#conv2d
//        output[b, i, j, k] =
//            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
//                filter[di, dj, q, k]
        
        for b in 0..<numBatches {
            // Accumulate index calculation
            let selfIndexB = b * imageHeight
            var pointerIndexI = b * numRows
            
            for i in 0..<numRows {
                for di in 0..<filterHeight {
                    let y = strides[1]*i+di - padTop
                    if(y<0){
                        continue
                    }
                    if(y>=imageHeight){
                        // If y is larger, it will never be smaller in this di loop.
                        break
                    }
                    // Accumulate index calculation
                    let selfIndexY = (selfIndexB + y) * imageWidth
                    let filterIndexDI = di * filterWidth
                    var pointerIndexJ = pointerIndexI * numCols
                    
                    for j in 0..<numCols {
                        // Can get pointer before calculate x.
                        let selfIndex = (selfIndexY + max(0, strides[2]*j - padLeft)) * numInChannels
                        var selfPointer = UnsafeMutablePointer<Element>(self.elements) + selfIndex
                        
                        for dj in 0..<filterWidth {
                            let x = strides[2]*j+dj - padLeft
                            if(x < 0 || x>=imageWidth){
                                continue
                            }
                            // Pointer of filter
                            let filterIndex = (filterIndexDI + dj) * numInChannels * numOutChannels
                            var filterPointer = UnsafeMutablePointer<Element>(filter.elements) + filterIndex
                            for _ in 0..<numInChannels { // Loop of q
                                // Pointer of elements
                                var pointer = UnsafeMutablePointer<Element>(z.elements) + pointerIndexJ * numOutChannels
                                for _ in 0..<numOutChannels { // Loop of k
                                    pointer.memory += selfPointer.memory * filterPointer.memory
                                    // Increment by k's grow
                                    pointer += 1
                                    filterPointer += 1
                                }
                                // Increment by q's grow
                                selfPointer += 1
                            }
                        }
                        pointerIndexJ += 1
                    }
                }
                pointerIndexI += 1
            }
        }
        
        return z
    }
    
    public func conv2d_fast(filter filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        
        // delete when batch num is
        assert(self.shape.dimensions[0] == 1, "Number of image batches must be 1")
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numOutChannels = filter.shape.dimensions[3].value
        
        let padAlongHeight = (numRows - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        let imageWidth = shape.dimensions[2].value
        let imageHeight = shape.dimensions[1].value
        let numInChannels = shape.dimensions[3].value
        
        let filterWidth = filter.shape.dimensions[1].value
        let filterHeight = filter.shape.dimensions[0].value
        
        // X shape = [ numRows*numCols , rowSize ]
        let rowSize = filterHeight * filterWidth * numInChannels
        let X = [Float](count: numRows * numCols * rowSize, repeatedValue: 0)
        var X_pointer = UnsafeMutablePointer<Float>(X)
        for y in 0..<numRows {
            for x in 0..<numCols{
                // Add (x,y)'s patch as a vector
                for c in 0..<filterHeight{ // Row number of patch
                    
                    let inputY = y*strides[1] + c - padTop // y cood in original image
                    if(inputY < 0 || inputY >= imageHeight){
                        X_pointer += filterWidth * numInChannels
                        continue
                    }
                    
                    var inputX = x*strides[2] - padLeft // x cood in image
                    
                    var startIndex = 0 // Relative index of starting data
                    var pixelCount = filterWidth // Number of pixels to copy
                    
                    if(inputX < 0){
                        // Shift startIndex if inputX is not in image
                        startIndex += -inputX * numInChannels
                        pixelCount -= -inputX
                        inputX = 0
                    }
                    if(inputX + pixelCount > imageWidth){
                        // Decrement pixelCount if end of data is not in image
                        pixelCount -= (inputX + pixelCount - imageWidth)
                    }
                    
                    let imageStartIndex = ((inputY * imageWidth) + inputX) * numInChannels
                    
                    let source = UnsafePointer<Float>(self.elements) + imageStartIndex
                    X_pointer += startIndex
                    memcpy(X_pointer, source, pixelCount * numInChannels * sizeof(Float))
                    X_pointer += filterWidth * numInChannels - startIndex
                }
            }
        }
        
        let a = UnsafePointer<Float>(X)
        let b = UnsafePointer<Float>(filter.elements)
        
        let z = Tensor(shape: [Dimension(numBatches), Dimension(numRows), Dimension(numCols), Dimension(numOutChannels)])
        
        let c = UnsafeMutablePointer<Float>(z.elements)
        
        let M = numRows * numCols
        let N = numOutChannels
        let K = rowSize
        
        
        // Calculate [M, N] matrix, it automatically turns into [numBatches, numRows, numCols, numOutChannels] Tensor
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K), 1.0,
                    a, Int32(K),
                    b, Int32(N), 1.0,
                    c, Int32(N))
        
        return z
    }
}