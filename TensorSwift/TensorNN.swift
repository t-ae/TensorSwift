import Darwin

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
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(ksize.count >= 4, "`ksize.count` must be greater than or equal to 4: \(ksize.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        assert(ksize[0] == 1 ,"`ksize[0]` must be 1")
        assert(ksize[3] == 1 ,"`ksize[3]` must be 1")
        
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numCols = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numRows = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numChannels = Int(ceil(Float(shape.dimensions[3].value) / Float(strides[3])))
        
        let padAlongHeight = (numCols - 1) * strides[1] + ksize[1] - shape.dimensions[1].value
        let padAlongWidth = (numRows - 1) * strides[2] + ksize[2] - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padBottom = padAlongHeight - padTop
        let padLeft = padAlongWidth / 2
        let padRight = padAlongWidth - padLeft
        
        var elements: [Element] = []
        elements.reserveCapacity(numBatches * numCols * numRows * numChannels)
        
        for batch in 0.stride(to: shape.dimensions[0].value, by: strides[0]) {
            var es: [Element] = Array.init(count: shape.dimensions[3].value, repeatedValue: 0)
            for y in (0-padTop).stride(to: shape.dimensions[1].value+padBottom-ksize[1]+1, by: strides[1]) {
                for x in (0-padLeft).stride(to: shape.dimensions[2].value+padRight-ksize[2]+1, by: strides[2]) {
                    for j in 0..<ksize[1] {
                        if y+j < 0 || y+j >= shape.dimensions[1].value {
                            continue
                        }
                        for i in 0..<ksize[2] {
                            if x+i < 0 || x+i >= shape.dimensions[2].value {
                                continue
                            }
                            es = es.enumerate().map { $0.element < self[batch, y+j, x+i, $0.index] ? self[batch, y+j, x+i, $0.index] : $0.element }
                        }
                    }
                    for e in es {
                        elements.append(e)
                    }
                }
            }
        }

        return Tensor(shape: [Dimension(numBatches) ,Dimension(numCols), Dimension(numRows), Dimension(numChannels)], elements: elements)
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
        
        let elements = [Element](count: numBatches * numCols * numRows * numOutChannels, repeatedValue: 0)
        
//      https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#conv2d
//        output[b, i, j, k] =
//            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
//                filter[di, dj, q, k]
        
        
        for b in 0..<numBatches {
            // インデックス計算を途中までaccumulate
            let selfIndexB = b * shape.dimensions[1].value
            var pointerIndexI = b * numRows
            
            for i in 0..<numRows {
                for di in 0..<filter.shape.dimensions[0].value { // filter height
                    let y = strides[1]*i+di - padTop
                    if(y<0){
                        continue
                    }
                    if(y>=self.shape.dimensions[1].value){
                        // Yが大きい場合それ以降も常に大きいのでbreak
                        break
                    }
                    // インデックス計算を途中までaccumulate
                    let selfIndexY = (selfIndexB + y) * shape.dimensions[2].value
                    let filterIndexDI = di * filter.shape.dimensions[1].value
                    var pointerIndexJ = pointerIndexI * numCols
                    
                    for j in 0..<numCols {
                        // xが確定する前にポインタを作れる(=xもシーケンシャルアクセス)
                        let selfIndex = (selfIndexY + max(0, strides[2]*j - padLeft)) * shape.dimensions[3].value
                        var selfPointer = UnsafeMutablePointer<Element>(self.elements) + selfIndex
                        
                        for dj in 0..<filter.shape.dimensions[1].value { // filter width
                            let x = strides[2]*j+dj - padLeft
                            if(x < 0 || x>=self.shape.dimensions[2].value){
                                continue
                            }
                            // filterのポインタ
                            let filterIndex = (filterIndexDI + dj) * filter.shape.dimensions[2].value * filter.shape.dimensions[3].value
                            var filterPointer = UnsafeMutablePointer<Element>(filter.elements) + filterIndex
                            for _ in 0..<filter.shape.dimensions[2].value { // in channelss (loop of q)
                                // elementsのポインタ
                                var pointer = UnsafeMutablePointer<Element>(elements) + pointerIndexJ * numOutChannels
                                for _ in 0..<numOutChannels { // loop of k
                                    pointer.memory += selfPointer.memory * filterPointer.memory
                                    // kの増加でインクリメント
                                    pointer += 1
                                    filterPointer += 1
                                }
                                // qの増加でインクリメント
                                selfPointer += 1
                            }
                        }
                        pointerIndexJ += 1
                    }
                }
                pointerIndexI += 1
            }
        }
        
        return Tensor(shape: [Dimension(numBatches) ,Dimension(numRows), Dimension(numCols), Dimension(numOutChannels)], elements: elements)
    }
}