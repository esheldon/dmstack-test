# dmstack-test
Code to test dm stack outputs using simulations

## Examples

```bash
# using the command line
dmstack-test --seed 3433 --output test.fits --ntrial 10 --gal-type dev --gal-hlr 0.5 --ngmix-model dev

dmstack-test-plots --flist test*.fits --output fracdiff-stars.png
```
