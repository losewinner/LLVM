// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -verify %s

@protocol NSObject
+alloc;
-init;
@end

@interface NSObject <NSObject> {}
@end

@interface BaseClass : NSObject
@end

@interface DerivedClass : BaseClass
-(void)testCasts:(BaseClass*)base;
@end

@implementation DerivedClass
-(void)testCasts:(BaseClass*)base {
  DerivedClass *derived = (DerivedClass*)base;
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass *' to derived type 'DerivedClass *'}}
  DerivedClass *derived_static = static_cast<DerivedClass*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass *' to derived type 'DerivedClass *'}}
  DerivedClass *derived_reinterpret = reinterpret_cast<DerivedClass*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass *' to derived type 'DerivedClass *'}}
  base = (BaseClass*)derived;  // no warning
  base = (BaseClass*)base;  // no warning
}
@end

template <typename T>
class WrappedObject
{
public:
  T get() const { return mMetalObject; }
  T mMetalObject = nullptr;
};

@protocol MTLCommandEncoder
@end
@protocol MTLRenderCommandEncoder
@end
class CommandEncoder : public WrappedObject<id<MTLCommandEncoder>> { };

class RenderCommandEncoder final : public CommandEncoder
{
private:
    // Override CommandEncoder
    id<MTLRenderCommandEncoder> get()
    {
        return static_cast<id<MTLRenderCommandEncoder>>(CommandEncoder::get());
    }
};
