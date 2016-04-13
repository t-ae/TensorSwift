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
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numChannels = Int(ceil(Float(shape.dimensions[3].value) / Float(strides[3])))
        
        let padAlongHeight = (numRows - 1) * strides[1] + ksize[1] - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + ksize[2] - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        // Initialize with -infinity for maximization.
        let elements = [Element](count: numBatches * numCols * numRows * numChannels, repeatedValue: -Float.infinity)
        
        for b in 0..<numBatches {
            var elementIndexI = b * numRows
            for i in 0..<numRows {
                for di in 0..<ksize[1] {
                    let y = i+di - padTop
                    if(y<0){
                        continue
                    }
                    if(y>=self.shape.dimensions[1].value){
                        break
                    }
                    var elementIndexJ = elementIndexI * numCols
                    for j in 0..<numCols {
                        var selfIndex = b
                        selfIndex = selfIndex * self.shape.dimensions[1].value + y
                        selfIndex = selfIndex * self.shape.dimensions[2].value + max(0, j-padLeft)
                        selfIndex = selfIndex * self.shape.dimensions[3].value
                        var selfPointer = UnsafeMutablePointer<Element>(self.elements) + selfIndex
                        for dj in 0..<ksize[2] {
                            let x = j+dj - padLeft
                            if(x<0 || x>=self.shape.dimensions[2].value){
                                continue
                            }
                            var elementPointer = UnsafeMutablePointer<Element>(elements) + elementIndexJ * numChannels
                            for _ in 0..<numChannels {
                                elementPointer.memory = max(elementPointer.memory, selfPointer.memory)
                                elementPointer += 1
                                selfPointer += 1
                            }
                        }
                        elementIndexJ += 1
                    }
                }
                elementIndexI += 1
            }
        }
        
        return Tensor(shape: [Dimension(numBatches) ,Dimension(numRows), Dimension(numCols), Dimension(numChannels)], elements: elements)
    }
    
    public func conv2d(filter filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numCols = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numRows = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numChannels = filter.shape.dimensions[3].value
        
        let padAlongHeight = (numCols - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numRows - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padBottom = padAlongHeight - padTop
        let padLeft = padAlongWidth / 2
        let padRight = padAlongWidth - padLeft
        
        var elements: [Element] = []
        elements.reserveCapacity(numBatches * numCols * numRows * numChannels)
        
        for batch in 0.stride(to: shape.dimensions[0].value, by: strides[0]) {
            for y in (0-padTop).stride(to: shape.dimensions[1].value+padBottom-filter.shape.dimensions[0].value+1, by: strides[1]) {
                for x in (0-padLeft).stride(to: shape.dimensions[2].value+padRight-filter.shape.dimensions[1].value+1, by: strides[2]) {
                    for channel in 0..<filter.shape.dimensions[3].value {
                        var e: Element = 0
                        for j in 0..<filter.shape.dimensions[0].value {
                            if y+j < 0 || y+j >= shape.dimensions[1].value {
                                continue
                            }
                            
                            for i in 0..<filter.shape.dimensions[1].value {
                                if x+i < 0 || x+i >= shape.dimensions[2].value {
                                    continue
                                }
                                
                                for h in 0..<filter.shape.dimensions[2].value {
                                    e += self[batch, y + j, x + i, h] * filter[j, i, h, channel]
                                }
                            }
                        }
                        elements.append(e)
                    }
                }
            }
        }
        
        return Tensor(shape: [Dimension(numBatches) ,Dimension(numCols), Dimension(numRows), Dimension(numChannels)], elements: elements)
    }
}