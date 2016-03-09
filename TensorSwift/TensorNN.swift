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
        
        fatalError("Unimplemented yet.")
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
        let numChannels = filter.shape.dimensions[2].value
        
        let padAlongHeight = (numCols - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numRows - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padBottom = padAlongHeight - padTop
        let padLeft = padAlongWidth / 2
        let padRight = padAlongWidth - padLeft
        
        var elements: [Element] = []
        elements.reserveCapacity(numBatches * numCols * numRows * numChannels)
        
        var batch = 0
        var y = 0
        var x = 0
        
        for batch=0; batch<shape.dimensions[0].value; batch+=strides[0] {
            var es: [Element] = Array.init(count: shape.dimensions[3].value, repeatedValue: 0)
            for y=0-padTop; y+ksize[1]-1<shape.dimensions[1].value+padBottom; y+=strides[1] {
                for x=0-padLeft; x+ksize[2]-1<shape.dimensions[2].value+padRight; x+=strides[2] {
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
        
        
        
        fatalError("Unimplemented yet.")
    }
}