import Darwin

public func **(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs, rhs).map(powf))
}

extension Tensor {
    public var sin: Tensor {
        return Tensor(shape: shape, elements: _elements.map(sinf))
    }

    public var cos: Tensor {
        return Tensor(shape: shape, elements: _elements.map(cosf))
    }

    public var tan: Tensor {
        return Tensor(shape: shape, elements: _elements.map(tanf))
    }

    public var asin: Tensor {
        return Tensor(shape: shape, elements: _elements.map(asinf))
    }
    
    public var acos: Tensor {
        return Tensor(shape: shape, elements: _elements.map(acosf))
    }
    
    public var atan: Tensor {
        return Tensor(shape: shape, elements: _elements.map(atanf))
    }
    
    public var sinh: Tensor {
        return Tensor(shape: shape, elements: _elements.map(sinhf))
    }
    
    public var cosh: Tensor {
        return Tensor(shape: shape, elements: _elements.map(coshf))
    }
    
    public var tanh: Tensor {
        return Tensor(shape: shape, elements: _elements.map(tanhf))
    }
    
    public var exp: Tensor {
        return Tensor(shape: shape, elements: _elements.map(expf))
    }
    
    public var log: Tensor {
        return Tensor(shape: shape, elements: _elements.map(logf))
    }
    
    public var sqrt: Tensor {
        return Tensor(shape: shape, elements: _elements.map(sqrtf))
    }
    
    public var cbrt: Tensor {
        return Tensor(shape: shape, elements: _elements.map(cbrtf))
    }
}