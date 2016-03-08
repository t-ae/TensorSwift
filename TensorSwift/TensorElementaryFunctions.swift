import Darwin

public func **(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs.elements, rhs.elements).map(powf))
}

public func **(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { powf($0, rhs) })
}

extension Tensor {
    public var sin: Tensor {
        return Tensor(shape: shape, elements: elements.map(sinf))
    }

    public var cos: Tensor {
        return Tensor(shape: shape, elements: elements.map(cosf))
    }

    public var tan: Tensor {
        return Tensor(shape: shape, elements: elements.map(tanf))
    }

    public var asin: Tensor {
        return Tensor(shape: shape, elements: elements.map(asinf))
    }
    
    public var acos: Tensor {
        return Tensor(shape: shape, elements: elements.map(acosf))
    }
    
    public var atan: Tensor {
        return Tensor(shape: shape, elements: elements.map(atanf))
    }
    
    public var sinh: Tensor {
        return Tensor(shape: shape, elements: elements.map(sinhf))
    }
    
    public var cosh: Tensor {
        return Tensor(shape: shape, elements: elements.map(coshf))
    }
    
    public var tanh: Tensor {
        return Tensor(shape: shape, elements: elements.map(tanhf))
    }
    
    public var exp: Tensor {
        return Tensor(shape: shape, elements: elements.map(expf))
    }
    
    public var log: Tensor {
        return Tensor(shape: shape, elements: elements.map(logf))
    }
    
    public var sqrt: Tensor {
        return Tensor(shape: shape, elements: elements.map(sqrtf))
    }
    
    public var cbrt: Tensor {
        return Tensor(shape: shape, elements: elements.map(cbrtf))
    }
}